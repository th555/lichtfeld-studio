/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core_new/tensor.hpp"
#include "lfs/kernels/bilateral_grid.cuh"
#include <memory>

namespace lfs::training {

    // Context for manual bilateral grid slice forward/backward
    // Optimized: store only minimal data (no tensor copies, no refcount overhead)
    struct BilateralGridSliceContext {
        const float* rgb_ptr;    // Pointer to RGB data [H, W, 3]
        int h, w;                // Image dimensions
        int image_idx;           // Image index for gradient accumulation
    };

    // Context for manual TV loss forward/backward
    // Optimized: empty context (grids are class members, no need to store)
    struct BilateralGridTVContext {
        // Empty - all data is in BilateralGrid class members
    };

    /**
     * @brief Bilateral grid for appearance modeling (LibTorch-free)
     *
     * Manages a learned 3D lookup table per image for appearance variations.
     * Grid coordinates are (x, y, grayscale) where grayscale = 0.299*R + 0.587*G + 0.114*B.
     */
    class BilateralGrid {
    public:
        /**
         * @brief Construct bilateral grid
         * @param num_images Number of images (grid batch size)
         * @param grid_W Grid width dimension
         * @param grid_H Grid height dimension
         * @param grid_L Grid depth/guidance dimension
         */
        BilateralGrid(int num_images, int grid_W = 16, int grid_H = 16, int grid_L = 8);

        /**
         * @brief Apply bilateral grid to image (manual forward, no autograd)
         * @param rgb Input image [H, W, 3]
         * @param image_idx Index of image (selects which grid to use)
         * @return (output_image, context) for backward pass
         */
        std::pair<lfs::core::Tensor, BilateralGridSliceContext> apply_forward(
            const lfs::core::Tensor& rgb, int image_idx);

        /**
         * @brief Backward pass for bilateral grid (manual gradients)
         * @param ctx Context from forward pass
         * @param grad_output Gradient w.r.t. output [H, W, 3]
         * @return grad_rgb - gradient w.r.t. input RGB
         * @note Gradients w.r.t. grid are accumulated into grids_grad_
         */
        lfs::core::Tensor apply_backward(
            const BilateralGridSliceContext& ctx,
            const lfs::core::Tensor& grad_output);

        /**
         * @brief Compute total variation loss (manual forward)
         * @return (loss_value, context) for backward pass
         */
        std::pair<float, BilateralGridTVContext> tv_loss_forward();

        /**
         * @brief Backward pass for TV loss (manual gradients)
         * @param ctx Context from forward pass
         * @param grad_loss Gradient of total loss w.r.t. TV loss (scalar)
         * @note Gradients are accumulated into grids_grad_
         */
        void tv_loss_backward(
            const BilateralGridTVContext& ctx,
            float grad_loss);

        /**
         * @brief Get grid parameters for optimizer
         * @return Reference to grids tensor [N, 12, L, H, W]
         */
        lfs::core::Tensor& parameters() { return grids_; }
        const lfs::core::Tensor& parameters() const { return grids_; }

        /**
         * @brief Get gradient buffer
         * @return Reference to gradient tensor [N, 12, L, H, W]
         */
        lfs::core::Tensor& grad() { return grids_grad_; }
        const lfs::core::Tensor& grad() const { return grids_grad_; }

        /**
         * @brief Zero gradients
         */
        void zero_grad();

        // Grid dimensions
        int grid_width() const { return grid_width_; }
        int grid_height() const { return grid_height_; }
        int grid_guidance() const { return grid_guidance_; }
        int num_images() const { return num_images_; }

    private:
        lfs::core::Tensor grids_;      // [N, 12, L, H, W] - grid parameters
        lfs::core::Tensor grids_grad_; // [N, 12, L, H, W] - accumulated gradients
        lfs::core::Tensor tv_temp_buffer_; // Temporary buffer for TV loss reduction

        int num_images_;
        int grid_width_;
        int grid_height_;
        int grid_guidance_;
    };

} // namespace lfs::training
