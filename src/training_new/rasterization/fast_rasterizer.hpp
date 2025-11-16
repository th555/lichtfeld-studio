/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/camera.hpp"
#include "core_new/splat_data.hpp"
#include "optimizer/render_output.hpp"
#include <rasterization_api.h>  // Use angle brackets to search include paths, not relative

namespace lfs::training {
    // Forward pass context - holds intermediate buffers needed for backward
    struct FastRasterizeContext {
        lfs::core::Tensor image;
        lfs::core::Tensor alpha;
        lfs::core::Tensor bg_color;  // Saved for alpha gradient computation

        // Gaussian parameters (saved to avoid re-fetching in backward)
        lfs::core::Tensor means;
        lfs::core::Tensor raw_scales;
        lfs::core::Tensor raw_rotations;
        lfs::core::Tensor shN;

        const float* w2c_ptr = nullptr;
        const float* cam_position_ptr = nullptr;

        // Forward context (contains buffer pointers, frame_id, etc.)
        fast_lfs::rasterization::ForwardContext forward_ctx;

        int active_sh_bases;
        int total_bases_sh_rest;
        int width;
        int height;
        float focal_x;
        float focal_y;
        float center_x;
        float center_y;
        float near_plane;
        float far_plane;
    };

    // Explicit forward pass - returns render output and context for backward
    std::pair<RenderOutput, FastRasterizeContext> fast_rasterize_forward(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color);

    // Explicit backward pass - computes gradients and accumulates them manually
    void fast_rasterize_backward(
        const FastRasterizeContext& ctx,
        const lfs::core::Tensor& grad_image,
        lfs::core::SplatData& gaussian_model);

    // Convenience wrapper for inference (no backward needed)
    inline RenderOutput fast_rasterize(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color) {
        auto [output, ctx] = fast_rasterize_forward(viewpoint_camera, gaussian_model, bg_color);
        return output;
    }
} // namespace lfs::training
