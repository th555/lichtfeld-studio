/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <torch/torch.h>

namespace gs {
    namespace bilateral_grid {

        // Low-level CUDA kernel wrappers (used by manual implementations)
        void slice_forward_cuda(
            const torch::Tensor& grid, // [12, L, H, W]
            const torch::Tensor& rgb,  // [H, W, 3]
            torch::Tensor& output,     // [H, W, 3]
            bool use_uniform_coords = true);

        std::tuple<torch::Tensor, torch::Tensor> slice_backward_cuda(
            const torch::Tensor& grid,       // [12, L, H, W]
            const torch::Tensor& rgb,        // [H, W, 3]
            const torch::Tensor& grad_output // [H, W, 3]
        );

        torch::Tensor tv_loss_forward_cuda(
            const torch::Tensor& grids // [N, 12, L, H, W]
        );

        torch::Tensor tv_loss_backward_cuda(
            const torch::Tensor& grids,      // [N, 12, L, H, W]
            const torch::Tensor& grad_output // scalar
        );

        // ============= MANUAL FORWARD/BACKWARD INTERFACE (no autograd) =============

        // Context for manual bilateral grid slice forward/backward
        struct BilateralGridSliceContext {
            torch::Tensor grid;  // [12, L, H, W]
            torch::Tensor rgb;   // [H, W, 3]
        };

        // Context for manual TV loss forward/backward
        struct BilateralGridTVContext {
            torch::Tensor grids; // [N, 12, L, H, W]
        };

        // Manual bilateral grid slice forward (no autograd)
        // Returns: (output_image, context)
        std::pair<torch::Tensor, BilateralGridSliceContext> bilateral_grid_slice_forward(
            const torch::Tensor& grid, // [12, L, H, W]
            const torch::Tensor& rgb   // [H, W, 3]
        );

        // Manual bilateral grid slice backward (no autograd)
        // Returns: (grad_grid, grad_rgb)
        std::tuple<torch::Tensor, torch::Tensor> bilateral_grid_slice_backward(
            const BilateralGridSliceContext& ctx,
            const torch::Tensor& grad_output // [H, W, 3] - gradient w.r.t. output
        );

        // Manual TV loss forward (no autograd)
        // Returns: (loss_value, context)
        std::pair<float, BilateralGridTVContext> bilateral_grid_tv_forward(
            const torch::Tensor& grids // [N, 12, L, H, W]
        );

        // Manual TV loss backward (no autograd)
        // Returns: grad_grids
        torch::Tensor bilateral_grid_tv_backward(
            const BilateralGridTVContext& ctx,
            float grad_loss // Gradient of loss w.r.t. TV loss value (scalar)
        );

    } // namespace bilateral_grid
} // namespace gs