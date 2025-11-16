/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "bilateral_grid.hpp"
#include "core_new/logger.hpp"
#include <stdexcept>

namespace lfs::training {

    BilateralGrid::BilateralGrid(int num_images, int grid_W, int grid_H, int grid_L)
        : num_images_(num_images),
          grid_width_(grid_W),
          grid_height_(grid_H),
          grid_guidance_(grid_L) {

        // Initialize grids directly with zeros (fused allocation + initialization)
        // TODO: Implement proper initialization with identity transform
        grids_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_images), 12,
                                           static_cast<size_t>(grid_L),
                                           static_cast<size_t>(grid_H),
                                           static_cast<size_t>(grid_W)},
                                          lfs::core::Device::CUDA,
                                          lfs::core::DataType::Float32);

        // Initialize gradient buffer (fused allocation + initialization)
        grids_grad_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_images), 12,
                                                static_cast<size_t>(grid_L),
                                                static_cast<size_t>(grid_H),
                                                static_cast<size_t>(grid_W)},
                                               lfs::core::Device::CUDA,
                                               lfs::core::DataType::Float32);

        // Allocate temporary buffer for TV loss reduction
        // Need max(2048, (N*L*H*W+255)/256) elements
        size_t total_elements = num_images * grid_L * grid_H * grid_W;
        size_t num_blocks = (total_elements + 255) / 256;
        size_t temp_size = std::max(size_t(2048), num_blocks);
        tv_temp_buffer_ = lfs::core::Tensor::empty({temp_size},
                                                   lfs::core::Device::CUDA,
                                                   lfs::core::DataType::Float32);

        LOG_DEBUG("BilateralGrid created: {} images, grid={}x{}x{}", num_images, grid_W, grid_H, grid_L);
    }

    std::pair<lfs::core::Tensor, BilateralGridSliceContext> BilateralGrid::apply_forward(
        const lfs::core::Tensor& rgb, int image_idx) {

        if (image_idx < 0 || image_idx >= num_images_) {
            throw std::out_of_range("Image index out of range");
        }

        // Get dimensions
        auto rgb_shape = rgb.shape();
        size_t h = rgb_shape[0];
        size_t w = rgb_shape[1];

        // Allocate output
        lfs::core::Tensor output = lfs::core::Tensor::empty({h, w, 3},
                                                              lfs::core::Device::CUDA,
                                                              lfs::core::DataType::Float32);

        // Calculate offset to grid for this image [N, 12, L, H, W]
        size_t grid_slice_size = 12 * grid_guidance_ * grid_height_ * grid_width_;
        float* grid_ptr = grids_.template ptr<float>() + (image_idx * grid_slice_size);

        // Call CUDA kernel
        kernels::launch_bilateral_grid_slice_forward(
            grid_ptr,
            rgb.template ptr<float>(),
            output.template ptr<float>(),
            grid_guidance_, grid_height_, grid_width_,
            h, w,
            nullptr);  // Default CUDA stream

        // Create minimal context for backward (no tensor copies, just pointers)
        BilateralGridSliceContext ctx{
            .rgb_ptr = rgb.template ptr<const float>(),
            .h = static_cast<int>(h),
            .w = static_cast<int>(w),
            .image_idx = image_idx};

        return {output, ctx};
    }

    lfs::core::Tensor BilateralGrid::apply_backward(
        const BilateralGridSliceContext& ctx,
        const lfs::core::Tensor& grad_output) {

        // Use dimensions from context (no tensor shape queries needed)
        size_t h = ctx.h;
        size_t w = ctx.w;

        // Allocate gradient output
        lfs::core::Tensor grad_rgb = lfs::core::Tensor::empty({h, w, 3},
                                                                lfs::core::Device::CUDA,
                                                                lfs::core::DataType::Float32);

        // Calculate offset to grid slice for this image [N, 12, L, H, W]
        size_t grid_slice_size = 12 * grid_guidance_ * grid_height_ * grid_width_;
        const float* grid_ptr = grids_.template ptr<const float>() + (ctx.image_idx * grid_slice_size);
        float* grad_grid_ptr = grids_grad_.template ptr<float>() + (ctx.image_idx * grid_slice_size);

        // Call CUDA kernel - writes gradients directly to grids_grad_ at correct offset
        // CUDA kernels use atomicAdd for gradient accumulation
        kernels::launch_bilateral_grid_slice_backward(
            grid_ptr,
            ctx.rgb_ptr,  // Use pointer from context (no tensor dereference)
            grad_output.template ptr<float>(),
            grad_grid_ptr,  // Write directly to offset in grids_grad_
            grad_rgb.template ptr<float>(),
            grid_guidance_, grid_height_, grid_width_,
            h, w,
            nullptr);  // Default CUDA stream

        return grad_rgb;
    }

    std::pair<float, BilateralGridTVContext> BilateralGrid::tv_loss_forward() {
        // Allocate output for loss (single scalar on device)
        lfs::core::Tensor tv_loss_device = lfs::core::Tensor::zeros({1},
                                                                      lfs::core::Device::CUDA,
                                                                      lfs::core::DataType::Float32);

        // Call CUDA kernel
        kernels::launch_bilateral_grid_tv_forward(
            grids_.template ptr<float>(),
            tv_loss_device.template ptr<float>(),
            tv_temp_buffer_.template ptr<float>(),
            num_images_, grid_guidance_, grid_height_, grid_width_,
            nullptr);  // Default CUDA stream

        // Copy loss to host
        float tv_loss_value = tv_loss_device.item<float>();

        // Return empty context (all data is in class members)
        return {tv_loss_value, {}};
    }

    void BilateralGrid::tv_loss_backward(
        const BilateralGridTVContext& /*ctx*/,  // Unused (context is empty)
        float grad_loss) {

        // Write gradients directly to grids_grad_ (no temp allocation needed)
        // CUDA kernel uses atomicAdd for accumulation
        kernels::launch_bilateral_grid_tv_backward(
            grids_.template ptr<float>(),        // Use class member directly
            grad_loss,
            grids_grad_.template ptr<float>(),  // Write directly to gradient buffer
            num_images_, grid_guidance_, grid_height_, grid_width_,
            nullptr);  // Default CUDA stream
    }

    void BilateralGrid::zero_grad() {
        grids_grad_.fill_(0.0f);
    }

} // namespace lfs::training
