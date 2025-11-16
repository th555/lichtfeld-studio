/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace lfs::training::kernels {

/**
 * @brief Fused kernel for computing grad_alpha in rasterization backward pass
 *
 * Computes: grad_alpha[h,w] = -sum_c(grad_image[...,c,...] * bg_color[c])
 *
 * This fuses multiply + sum + negate into a single kernel, avoiding:
 * - Multiple kernel launches (3-4 separate ops)
 * - Intermediate memory allocations
 * - Poor memory coalescing from segmented reduce
 *
 * Expected performance: 10-50x faster than separate tensor ops
 *
 * @param grad_image Input gradient [3,H,W] or [H,W,3]
 * @param bg_color Background color [3]
 * @param grad_alpha Output gradient [H,W]
 * @param H Height
 * @param W Width
 * @param is_chw_layout true if [3,H,W], false if [H,W,3]
 * @param stream CUDA stream
 */
void launch_fused_grad_alpha(
    const float* grad_image,
    const float* bg_color,
    float* grad_alpha,
    int H, int W,
    bool is_chw_layout,
    cudaStream_t stream = nullptr
);

/**
 * @brief Fused kernel for background blending in rasterization forward pass
 *
 * Computes: output[c,h,w] = image[c,h,w] + (1 - alpha[h,w]) * bg_color[c]
 *
 * This fuses 5 separate operations into a single kernel:
 * - alpha negation: -alpha
 * - scalar add: + 1.0
 * - bg_color broadcast multiply
 * - image + bg_contribution
 *
 * Expected performance: 5-10x faster than separate tensor ops
 *
 * @param image Raw rendered image [3,H,W]
 * @param alpha Alpha channel [1,H,W]
 * @param bg_color Background color [3]
 * @param output Output image [3,H,W]
 * @param H Height
 * @param W Width
 * @param stream CUDA stream
 */
void launch_fused_background_blend(
    const float* image,
    const float* alpha,
    const float* bg_color,
    float* output,
    int H, int W,
    cudaStream_t stream = nullptr
);

} // namespace lfs::training::kernels
