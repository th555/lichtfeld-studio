/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <torch/torch.h>

namespace gs::training {

    class BilateralGrid {
    public:
        BilateralGrid(int num_images, int grid_W = 16, int grid_H = 16, int grid_L = 8);

        // Apply bilateral grid to rendered image
        torch::Tensor apply(const torch::Tensor& rgb, int image_idx);

        // Compute total variation loss
        torch::Tensor tv_loss() const;

        // Get parameters for optimizer
        torch::Tensor parameters() { return grids_; }
        const torch::Tensor& parameters() const { return grids_; }

        // Grid dimensions
        int grid_width() const { return grid_width_; }
        int grid_height() const { return grid_height_; }
        int grid_guidance() const { return grid_guidance_; }

        // Test/comparison helpers - wrap around apply() to match new API signature
        // Returns (output, input_copy_with_grad) where input_copy_with_grad serves as context
        std::pair<torch::Tensor, torch::Tensor> apply_forward(const torch::Tensor& rgb, int image_idx) {
            auto rgb_with_grad = rgb.requires_grad_(true);
            auto output = apply(rgb_with_grad, image_idx);
            return {output, rgb_with_grad};  // Return output and input (serves as context for backward)
        }

        // Test/comparison helper for backward - uses PyTorch autograd
        torch::Tensor apply_backward(const torch::Tensor& rgb_with_grad, const torch::Tensor& grad_output, int image_idx) {
            // rgb_with_grad is the context from forward pass
            // We need to compute gradients w.r.t. the input
            if (rgb_with_grad.grad().defined()) {
                rgb_with_grad.grad().zero_();
            }

            // Get the output again (it should be in computation graph from forward)
            auto output = apply(rgb_with_grad, image_idx);
            output.backward(grad_output);

            return rgb_with_grad.grad();
        }

        // Test/comparison helpers for TV loss
        std::pair<torch::Tensor, int> tv_loss_forward() {
            auto loss = tv_loss();
            return {loss, 0};  // Return (loss, dummy_context)
        }

        void tv_loss_backward(int /*dummy_context*/, float grad_loss) {
            // TV loss backward is handled by PyTorch autograd
            // The grids_ tensor tracks gradients automatically
            auto loss = tv_loss();
            loss.backward(torch::tensor(grad_loss, loss.options()));
        }

    private:
        torch::Tensor grids_; // [N, 12, L, H, W]
        int num_images_;
        int grid_width_;
        int grid_height_;
        int grid_guidance_;
    };

} // namespace gs::training