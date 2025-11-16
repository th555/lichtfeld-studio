/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fast_rasterizer.hpp"
#include "core_new/logger.hpp"
#include "training_new/kernels/grad_alpha.hpp"

namespace lfs::training {
    std::pair<RenderOutput, FastRasterizeContext> fast_rasterize_forward(
        core::Camera& viewpoint_camera,
        core::SplatData& gaussian_model,
        core::Tensor& bg_color) {
        // Get camera parameters
        const int width = viewpoint_camera.image_width();
        const int height =viewpoint_camera.image_height();
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        // Get Gaussian parameters
        auto &means = gaussian_model.means();
        auto &raw_opacities = gaussian_model.opacity_raw();
        auto &raw_scales = gaussian_model.scaling_raw();
        auto &raw_rotations = gaussian_model.rotation_raw();
        auto &sh0 = gaussian_model.sh0();
        auto &shN = gaussian_model.shN();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        // Get direct GPU pointers (tensors are already contiguous on CUDA)
        const float* w2c_ptr = viewpoint_camera.world_view_transform_ptr();
        const float* cam_position_ptr = viewpoint_camera.cam_position_ptr();

        const int n_primitives = static_cast<int>(means.shape()[0]);
        const int total_bases_sh_rest = static_cast<int>(shN.shape()[1]);

        // Pre-allocate output tensors (reused across iterations)
        thread_local core::Tensor image;
        thread_local core::Tensor alpha;
        thread_local core::Tensor output_image;
        thread_local int last_width = -1;
        thread_local int last_height = -1;

        // Only reallocate if dimensions changed
        if (last_width != width || last_height != height) {
            image = core::Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)});
            alpha = core::Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)});
            output_image = core::Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)}, core::Device::CUDA);
            last_width = width;
            last_height = height;
        }

        // Call forward_raw with raw pointers (no PyTorch wrappers)
        auto forward_ctx = fast_lfs::rasterization::forward_raw(
            means.ptr<float>(),
            raw_scales.ptr<float>(),
            raw_rotations.ptr<float>(),
            raw_opacities.ptr<float>(),
            sh0.ptr<float>(),
            shN.ptr<float>(),
            w2c_ptr,
            cam_position_ptr,
            image.ptr<float>(),
            alpha.ptr<float>(),
            n_primitives,
            active_sh_bases,
            total_bases_sh_rest,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near_plane,
            far_plane);

        // Prepare render output
        RenderOutput render_output;
        // output = image + (1 - alpha) * bg_color
        // (output_image is pre-allocated above)

        kernels::launch_fused_background_blend(
            image.ptr<float>(),
            alpha.ptr<float>(),
            bg_color.ptr<float>(),
            output_image.ptr<float>(),
            height,
            width,
            nullptr  // default stream
        );

        render_output.image = output_image;
        render_output.alpha = alpha;
        render_output.width = width;
        render_output.height = height;

        // Prepare context for backward
        FastRasterizeContext ctx;
        ctx.image = image;
        ctx.alpha = alpha;
        ctx.bg_color = bg_color;  // Save bg_color for alpha gradient

        // Save parameters (avoid re-fetching in backward)
        ctx.means = means;
        ctx.raw_scales = raw_scales;
        ctx.raw_rotations = raw_rotations;
        ctx.shN = shN;

        // Store camera pointers directly (tensors are managed by camera, already contiguous)
        ctx.w2c_ptr = w2c_ptr;
        ctx.cam_position_ptr = cam_position_ptr;

        // Store forward context (contains buffer pointers, frame_id, etc.)
        ctx.forward_ctx = forward_ctx;

        ctx.active_sh_bases = active_sh_bases;
        ctx.total_bases_sh_rest = total_bases_sh_rest;
        ctx.width = width;
        ctx.height = height;
        ctx.focal_x = fx;
        ctx.focal_y = fy;
        ctx.center_x = cx;
        ctx.center_y = cy;
        ctx.near_plane = near_plane;
        ctx.far_plane = far_plane;

        return {render_output, ctx};
    }

    void fast_rasterize_backward(
        const FastRasterizeContext& ctx,
        const core::Tensor& grad_image,
        core::SplatData& gaussian_model) {

        // Compute gradient w.r.t. alpha from background blending
        // Forward: output_image = image + (1 - alpha) * bg_color
        // where bg_color is [3], alpha is [1, H, W], output_image is [3, H, W]
        //
        // Backward:
        // ∂L/∂image_raw = ∂L/∂output_image (grad_image)
        // ∂L/∂alpha = -sum_over_channels(∂L/∂output_image * bg_color)
        //
        // grad_image shape: [3, H, W] or [H, W, 3]
        // bg_color shape: [3]
        // alpha shape: [1, H, W]

        // Use fused kernel (4-5x faster than LibTorch, 18-252x faster than separate ops)
        int H, W;
        bool is_chw_layout;

        if (grad_image.shape()[0] == 3) {
            // Layout: [3, H, W]
            is_chw_layout = true;
            H = static_cast<int>(grad_image.shape()[1]);
            W = static_cast<int>(grad_image.shape()[2]);
        } else if (grad_image.shape()[2] == 3) {
            // Layout: [H, W, 3]
            is_chw_layout = false;
            H = static_cast<int>(grad_image.shape()[0]);
            W = static_cast<int>(grad_image.shape()[1]);
        } else {
            throw std::runtime_error("Unexpected grad_image shape in fast_rasterize_backward");
        }

        auto grad_alpha = core::Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, core::Device::CUDA);

        kernels::launch_fused_grad_alpha(
            grad_image.ptr<float>(),
            ctx.bg_color.ptr<float>(),
            grad_alpha.ptr<float>(),
            H, W,
            is_chw_layout,
            nullptr  // default stream
        );

        const int n_primitives = static_cast<int>(ctx.means.shape()[0]);

        // Call backward_raw with raw pointers to SplatData gradient buffers
        const bool update_densification_info = gaussian_model._densification_info.ndim() > 0 &&
                                                gaussian_model._densification_info.shape()[0] > 0;
        auto backward_result = fast_lfs::rasterization::backward_raw(
            update_densification_info ? gaussian_model._densification_info.ptr<float>() : nullptr,
            grad_image.ptr<float>(),
            grad_alpha.ptr<float>(),
            ctx.image.ptr<float>(),
            ctx.alpha.ptr<float>(),
            ctx.means.ptr<float>(),
            ctx.raw_scales.ptr<float>(),
            ctx.raw_rotations.ptr<float>(),
            ctx.shN.ptr<float>(),
            ctx.w2c_ptr,
            ctx.cam_position_ptr,
            ctx.forward_ctx,
            gaussian_model.means_grad().ptr<float>(),        // Direct access to SplatData gradients
            gaussian_model.scaling_grad().ptr<float>(),
            gaussian_model.rotation_grad().ptr<float>(),
            gaussian_model.opacity_grad().ptr<float>(),
            gaussian_model.sh0_grad().ptr<float>(),
            gaussian_model.shN_grad().ptr<float>(),
            nullptr,  // grad_w2c not needed for now
            n_primitives,
            ctx.active_sh_bases,
            ctx.total_bases_sh_rest,
            ctx.width,
            ctx.height,
            ctx.focal_x,
            ctx.focal_y,
            ctx.center_x,
            ctx.center_y);

        if (!backward_result.success) {
            throw std::runtime_error(std::string("Backward failed: ") + backward_result.error_message);
        }

        // Gradients are ACCUMULATED (+=) directly into SplatData buffers
        // This allows multiple loss terms (photometric, regularization, etc.) to contribute
        // The trainer calls zero_gradients() at the END of strategy->step() (after optimizer update)
    }
} // namespace lfs::training
