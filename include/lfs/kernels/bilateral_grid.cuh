/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>
#include <utility>

namespace lfs::training::kernels {

/**
 * @brief Forward pass for bilateral grid slicing (LibTorch-free)
 *
 * Applies a learned 3D lookup table (bilateral grid) to an image using trilinear interpolation.
 * Grid coordinates are computed from (x, y, grayscale) where grayscale = 0.299*R + 0.587*G + 0.114*B.
 *
 * @param grid Input bilateral grid [12, L, H, W] - 12 coefficients for 3×4 affine transform
 * @param rgb Input RGB image [h, w, 3]
 * @param output Output RGB image [h, w, 3]
 * @param L Grid depth dimension
 * @param H Grid height dimension
 * @param W Grid width dimension
 * @param h Image height
 * @param w Image width
 * @param stream CUDA stream
 */
void launch_bilateral_grid_slice_forward(
    const float* grid,
    const float* rgb,
    float* output,
    int L, int H, int W,
    int h, int w,
    cudaStream_t stream = nullptr);

/**
 * @brief Backward pass for bilateral grid slicing (LibTorch-free)
 *
 * Computes gradients w.r.t. grid and RGB for bilateral grid slice operation.
 *
 * @param grid Input bilateral grid [12, L, H, W]
 * @param rgb Input RGB image [h, w, 3]
 * @param grad_output Gradient w.r.t. output [h, w, 3]
 * @param grad_grid Output gradient w.r.t. grid [12, L, H, W] (accumulated)
 * @param grad_rgb Output gradient w.r.t. RGB [h, w, 3]
 * @param L Grid depth dimension
 * @param H Grid height dimension
 * @param W Grid width dimension
 * @param h Image height
 * @param w Image width
 * @param stream CUDA stream
 */
void launch_bilateral_grid_slice_backward(
    const float* grid,
    const float* rgb,
    const float* grad_output,
    float* grad_grid,
    float* grad_rgb,
    int L, int H, int W,
    int h, int w,
    cudaStream_t stream = nullptr);

/**
 * @brief Forward pass for bilateral grid total variation loss (LibTorch-free)
 *
 * Computes total variation regularization on the bilateral grid:
 * TV = mean(sum of squared differences between neighboring cells in x, y, z directions)
 *
 * @param grids Input bilateral grids [N, 12, L, H, W]
 * @param tv_loss Output scalar loss (1 element, on device)
 * @param temp_buffer Temporary buffer for partial sums (min(2048, (N*L*H*W+255)/256) elements)
 * @param N Number of images
 * @param L Grid depth dimension
 * @param H Grid height dimension
 * @param W Grid width dimension
 * @param stream CUDA stream
 */
void launch_bilateral_grid_tv_forward(
    const float* grids,
    float* tv_loss,
    float* temp_buffer,
    int N, int L, int H, int W,
    cudaStream_t stream = nullptr);

/**
 * @brief Backward pass for bilateral grid total variation loss (LibTorch-free)
 *
 * Computes gradients w.r.t. grids for total variation loss.
 *
 * @param grids Input bilateral grids [N, 12, L, H, W]
 * @param grad_output Gradient w.r.t. scalar loss (scalar)
 * @param grad_grids Output gradient w.r.t. grids [N, 12, L, H, W]
 * @param N Number of images
 * @param L Grid depth dimension
 * @param H Grid height dimension
 * @param W Grid width dimension
 * @param stream CUDA stream
 */
void launch_bilateral_grid_tv_backward(
    const float* grids,
    float grad_output,
    float* grad_grids,
    int N, int L, int H, int W,
    cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
