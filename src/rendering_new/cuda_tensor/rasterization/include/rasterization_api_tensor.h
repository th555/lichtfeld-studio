/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/tensor.hpp"
#include <tuple>

namespace lfs::rendering {

    // Import Tensor from lfs::core
    using lfs::core::Tensor;

    /**
     * @brief Forward rasterization with custom Tensor types (libtorch-free)
     *
     * @param means Gaussian means [N, 3]
     * @param scales_raw Gaussian scales in log-space [N, 3]
     * @param rotations_raw Gaussian rotations (unnormalized quaternions) [N, 4]
     * @param opacities_raw Gaussian opacities in logit-space [N, 1]
     * @param sh_coefficients_0 SH coefficients degree 0 [N, 1, 3]
     * @param sh_coefficients_rest SH coefficients degree 1+ [N, (degree+1)²-1, 3]
     * @param w2c World-to-camera transform matrix [4, 4]
     * @param cam_position Camera position in world space [3]
     * @param active_sh_bases Number of active SH bases (degree+1)²
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param focal_x Focal length in x direction
     * @param focal_y Focal length in y direction
     * @param center_x Principal point x coordinate
     * @param center_y Principal point y coordinate
     * @param near_plane Near clipping plane
     * @param far_plane Far clipping plane
     *
     * @return Tuple of (rendered_image [3, H, W], alpha_map [1, H, W])
     */
    std::tuple<Tensor, Tensor>
    forward_wrapper_tensor(
        const Tensor& means,
        const Tensor& scales_raw,
        const Tensor& rotations_raw,
        const Tensor& opacities_raw,
        const Tensor& sh_coefficients_0,
        const Tensor& sh_coefficients_rest,
        const Tensor& w2c,
        const Tensor& cam_position,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane);

} // namespace lfs::rendering
