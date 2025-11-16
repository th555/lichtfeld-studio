/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/bilateral_grid.cuh"
#include "lfs/core/memory_ops.cuh"
#include <cuda_runtime.h>

namespace lfs::training::kernels {

using namespace lfs::core;

__global__ void bilateral_grid_slice_backward_kernel(
    const float* __restrict__ grid,        // [12, L, H, W]
    const float* __restrict__ rgb,         // [h, w, 3]
    const float* __restrict__ grad_output, // [h, w, 3]
    float* __restrict__ grad_grid,         // [12, L, H, W]
    float* __restrict__ grad_rgb,          // [h, w, 3]
    int L, int H, int W,
    int h, int w) {

    // Advanced indexing to reduce atomicAdd conflicts while maintaining accuracy
    // This redistributes threads to spread out atomic operations better
    int wi = threadIdx.x * ((w + blockDim.x - 1) / blockDim.x) + blockIdx.x;
    int hi = threadIdx.y * ((h + blockDim.y - 1) / blockDim.y) + blockIdx.y;

    if (wi >= w || hi >= h)
        return;

    int pixel_idx = hi * w + wi;
    int rgb_offset = pixel_idx * 3;

    // Vectorized RGB load with safety checks
    RGB rgb_val = load_rgb_cs(&rgb[rgb_offset]);
    float sr = isfinite(rgb_val.r) ? rgb_val.r : 0.5f;
    float sg = isfinite(rgb_val.g) ? rgb_val.g : 0.5f;
    float sb = isfinite(rgb_val.b) ? rgb_val.b : 0.5f;

    // Grid coordinates (uniform sampling)
    float x = (float)wi / (float)(w - 1) * (W - 1);
    float y = (float)hi / (float)(h - 1) * (H - 1);
    float z = (kC2G_r * sr + kC2G_g * sg + kC2G_b * sb) * (L - 1);

    // Floor + ceil, clamped
    int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);
    int z1 = z0 + 1;
    z0 = min(max(z0, 0), L - 1);
    z1 = min(max(z1, 0), L - 1);

    // Fractional parts
    float fx = x - x0, fy = y - y0, fz = z - z0;

    // Vectorized gradient load with safety checks
    RGB grad = load_rgb_cs(&grad_output[rgb_offset]);
    float dr = isfinite(grad.r) ? grad.r : 0.0f;
    float dg = isfinite(grad.g) ? grad.g : 0.0f;
    float db = isfinite(grad.b) ? grad.b : 0.0f;
    float vr = 0.0f, vg = 0.0f, vb = 0.0f;

    // Precompute interpolation weights
    float w000 = (1 - fx) * (1 - fy) * (1 - fz);
    float w001 = fx * (1 - fy) * (1 - fz);
    float w010 = (1 - fx) * fy * (1 - fz);
    float w011 = fx * fy * (1 - fz);
    float w100 = (1 - fx) * (1 - fy) * fz;
    float w101 = fx * (1 - fy) * fz;
    float w110 = (1 - fx) * fy * fz;
    float w111 = fx * fy * fz;

    float gz_grad = 0.0f;

    // Process all 8 corners
#pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        int xi = (corner & 1) ? x1 : x0;
        int yi = (corner & 2) ? y1 : y0;
        int zi = (corner & 4) ? z1 : z0;

        // Select weight based on corner
        float w = (corner == 0) ? w000 : (corner == 1) ? w001 :
                  (corner == 2) ? w010 : (corner == 3) ? w011 :
                  (corner == 4) ? w100 : (corner == 5) ? w101 :
                  (corner == 6) ? w110 : w111;

        // Derivative w.r.t. z for discontinuity gradient
        float dfdz = ((corner & 1) ? fx : (1 - fx)) *
                     ((corner & 2) ? fy : (1 - fy)) *
                     ((corner & 4) ? 1 : -1);

        float trilerp = 0.0f;

#pragma unroll
        for (int ci = 0; ci < 12; ++ci) {
            int grid_idx = (ci * L + zi) * H * W + yi * W + xi;
            int si = ci % 4, di = ci / 4;

            float r_coeff = (si == 0 ? sr : si == 1 ? sg : si == 2 ? sb : 1.0f);
            float gout = (di == 0 ? dr : di == 1 ? dg : db);

            // Read grid value with read-only cache hint
            float v = load_ro(&grid[grid_idx]);

            // Accumulate RGB gradients
            if (si < 3)
                (si == 0 ? vr : si == 1 ? vg : vb) += v * w * gout;

            float grad_weight = r_coeff * gout;
            trilerp += v * grad_weight;

            // Accumulate grid gradients (direct atomic)
            atomicAdd(grad_grid + grid_idx, w * grad_weight);
        }
        gz_grad += dfdz * (L - 1) * trilerp;
    }

    // Apply discontinuity masking and save gradients
    gz_grad *= (float)(z0 != z && z1 != z);
    grad_rgb[rgb_offset + 0] = vr + kC2G_r * gz_grad;
    grad_rgb[rgb_offset + 1] = vg + kC2G_g * gz_grad;
    grad_rgb[rgb_offset + 2] = vb + kC2G_b * gz_grad;
}

void launch_bilateral_grid_slice_backward(
    const float* grid,
    const float* rgb,
    const float* grad_output,
    float* grad_grid,
    float* grad_rgb,
    int L, int H, int W,
    int h, int w,
    cudaStream_t stream) {

    // Use 2D grid to reduce atomic conflicts
    dim3 block(16, 16);
    dim3 grid_dim(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);

    bilateral_grid_slice_backward_kernel<<<grid_dim, block, 0, stream>>>(
        grid, rgb, grad_output, grad_grid, grad_rgb, L, H, W, h, w);
}

} // namespace lfs::training::kernels
