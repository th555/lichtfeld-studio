/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core_new/tensor.hpp"
#include <tuple>
#include <vector>

namespace lfs::training::kernels {

// Pre-allocated workspace for SSIM computation (eliminates 120GB allocation churn)
struct SSIMWorkspace {
    // Forward pass buffers
    lfs::core::Tensor ssim_map;         // [N, C, H, W]
    lfs::core::Tensor dm_dmu1;          // [N, C, H, W]
    lfs::core::Tensor dm_dsigma1_sq;    // [N, C, H, W]
    lfs::core::Tensor dm_dsigma12;      // [N, C, H, W]

    // Backward pass buffers
    lfs::core::Tensor dL_dmap;          // [N, C, H, W]
    lfs::core::Tensor dL_dimg1;         // [N, C, H, W]

    // Cropped buffer for efficient mean computation (avoids .contiguous() allocation)
    lfs::core::Tensor ssim_map_cropped; // [N, C, H-10, W-10] contiguous buffer

    // Track allocated size
    std::vector<size_t> allocated_shape;

    // Resize workspace if needed (only reallocates if shape changed)
    void ensure_size(const std::vector<size_t>& shape) {
        if (allocated_shape != shape) {
            lfs::core::TensorShape tshape(shape);
            ssim_map = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            dL_dmap = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            dL_dimg1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);

            // Allocate cropped buffer (10 pixels smaller in H and W for valid padding)
            if (shape.size() == 4 && shape[2] > 10 && shape[3] > 10) {
                std::vector<size_t> cropped_shape = {shape[0], shape[1], shape[2] - 10, shape[3] - 10};
                ssim_map_cropped = lfs::core::Tensor::empty(lfs::core::TensorShape(cropped_shape), lfs::core::Device::CUDA);
            }

            allocated_shape = shape;
        }
    }
};

// Context for manual SSIM forward/backward (like RasterizeContext)
struct SSIMContext {
    lfs::core::Tensor img1;
    lfs::core::Tensor img2;
    lfs::core::Tensor dm_dmu1;
    lfs::core::Tensor dm_dsigma1_sq;
    lfs::core::Tensor dm_dsigma12;
    int original_h;
    int original_w;
    bool apply_valid_padding;
};

std::pair<lfs::core::Tensor, SSIMContext> ssim_forward(
    const lfs::core::Tensor& img1,
    const lfs::core::Tensor& img2,
    bool apply_valid_padding = true);

// Optimized version with pre-allocated workspace (eliminates allocation churn)
std::pair<lfs::core::Tensor, SSIMContext> ssim_forward(
    const lfs::core::Tensor& img1,
    const lfs::core::Tensor& img2,
    SSIMWorkspace& workspace,
    bool apply_valid_padding = true);

// Manual SSIM backward (no autograd) - computes gradient w.r.t. img1
lfs::core::Tensor ssim_backward(
    const SSIMContext& ctx,
    float grad_loss);  // Gradient of loss w.r.t. SSIM value (scalar)

// Optimized version with pre-allocated workspace
lfs::core::Tensor ssim_backward(
    const SSIMContext& ctx,
    SSIMWorkspace& workspace,
    float grad_loss);

} // namespace lfs::training::kernels
