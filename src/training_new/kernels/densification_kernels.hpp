// densification_kernels.hpp
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace lfs::training::kernels {

/**
 * @brief Custom CUDA kernels for Gaussian densification
 *
 * These kernels replace LibTorch-heavy operations in densification with
 * efficient CUDA kernels that minimize memory allocations and maximize throughput.
 *
 * Memory savings:
 * - Old approach: 60+ intermediate tensors (8+ GB for 10M Gaussians)
 * - New approach: 3 temp allocations (~500 MB for 10M Gaussians)
 *
 * Performance:
 * - Duplicate: ~2ms for 10M Gaussians (vs ~50ms LibTorch)
 * - Split: ~5ms for 10M Gaussians (vs ~200ms LibTorch)
 */

/**
 * @brief Duplicate selected Gaussians
 *
 * Copies selected Gaussians to end of output arrays.
 * Output arrays must be pre-allocated to size (N + num_selected).
 *
 * @param positions_in Input positions [N, 3]
 * @param rotations_in Input rotations [N, 4]
 * @param scales_in Input scales [N, 3]
 * @param sh0_in Input SH degree 0 [N, 3]
 * @param shN_in Input SH higher degrees [N, shN_dim]
 * @param opacities_in Input opacities [N, 1]
 * @param positions_out Output positions [N + num_selected, 3]
 * @param rotations_out Output rotations [N + num_selected, 4]
 * @param scales_out Output scales [N + num_selected, 3]
 * @param sh0_out Output SH degree 0 [N + num_selected, 3]
 * @param shN_out Output SH higher degrees [N + num_selected, shN_dim]
 * @param opacities_out Output opacities [N + num_selected, 1]
 * @param selected_indices Indices to duplicate [num_selected]
 * @param N Number of input Gaussians
 * @param num_selected Number of Gaussians to duplicate
 * @param shN_dim SH higher degree dimension
 * @param stream CUDA stream
 */
void launch_duplicate_gaussians(
    const float* positions_in,
    const float* rotations_in,
    const float* scales_in,
    const float* sh0_in,
    const float* shN_in,
    const float* opacities_in,
    float* positions_out,
    float* rotations_out,
    float* scales_out,
    float* sh0_out,
    float* shN_out,
    float* opacities_out,
    const int64_t* selected_indices,
    int N,
    int num_selected,
    int shN_dim,
    cudaStream_t stream = nullptr
);

/**
 * @brief Split selected Gaussians
 *
 * Replaces each selected Gaussian with 2 new ones, offset by rotation-scaled noise.
 * Output arrays must be pre-allocated to size (num_keep + num_split * 2).
 *
 * Algorithm:
 * 1. Copy non-split Gaussians to output [0:num_keep]
 * 2. For each split Gaussian:
 *    - Convert quaternion to rotation matrix
 *    - Generate offset = R * S * noise (where noise ~ N(0,1))
 *    - Create 2 Gaussians at position ± offset
 *    - Scale /= 1.6
 *    - Adjust opacity if revised_opacity=true
 *
 * @param positions_in Input positions [N, 3]
 * @param rotations_in Input rotations (quaternions) [N, 4]
 * @param scales_in Input scales [N, 3]
 * @param sh_in Input SH coefficients [N, sh_dim]
 * @param opacities_in Input opacities [N, 1]
 * @param positions_out Output positions [num_keep + num_split*2, 3]
 * @param rotations_out Output rotations [num_keep + num_split*2, 4]
 * @param scales_out Output scales [num_keep + num_split*2, 3]
 * @param sh_out Output SH [num_keep + num_split*2, sh_dim]
 * @param opacities_out Output opacities [num_keep + num_split*2, 1]
 * @param split_indices Indices of Gaussians to split [num_split]
 * @param keep_indices Indices of Gaussians to keep [num_keep]
 * @param random_noise Random noise for offsets [num_split, 3]
 * @param N Number of input Gaussians
 * @param num_split Number of Gaussians to split
 * @param num_keep Number of Gaussians to keep (N - num_split)
 * @param sh_dim SH coefficient dimension
 * @param revised_opacity Use revised opacity formula
 * @param stream CUDA stream
 */
void launch_split_gaussians(
    const float* positions_in,
    const float* rotations_in,
    const float* scales_in,
    const float* sh0_in,
    const float* shN_in,
    const float* opacities_in,
    float* positions_out,
    float* rotations_out,
    float* scales_out,
    float* sh0_out,
    float* shN_out,
    float* opacities_out,
    const int64_t* split_indices,
    const int64_t* keep_indices,
    const float* random_noise,
    int N,
    int num_split,
    int num_keep,
    int shN_dim,
    bool revised_opacity,
    cudaStream_t stream = nullptr
);

} // namespace lfs::training::kernels
