/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/bilateral_grid.cuh"
#include "lfs/core/memory_ops.cuh"
#include <cuda_runtime.h>

namespace lfs::training::kernels {

using namespace lfs::core;

__global__ void bilateral_grid_slice_forward_kernel(
    const float* __restrict__ grid, // [12, L, H, W]
    const float* __restrict__ rgb,  // [h, w, 3]
    float* __restrict__ output,     // [h, w, 3]
    int L, int H, int W,
    int h, int w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= h * w)
        return;

    int hi = idx / w;
    int wi = idx % w;

    // Vectorized RGB load with streaming cache hint
    int rgb_idx = idx * 3;
    RGB color = load_rgb_cs(&rgb[rgb_idx]);
    float sr = color.r;
    float sg = color.g;
    float sb = color.b;
    float dr = 0.0f, dg = 0.0f, db = 0.0f;

    // Compute grid coordinates (uniform sampling)
    float gx = (float)wi / (float)(w - 1);
    float gy = (float)hi / (float)(h - 1);
    float gz = kC2G_r * sr + kC2G_g * sg + kC2G_b * sb;

    float x = gx * (W - 1);
    float y = gy * (H - 1);
    float z = gz * (L - 1);

    // Trilinear interpolation setup
    int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);
    int z1 = z0 + 1;
    z0 = min(max(z0, 0), L - 1);
    z1 = min(max(z1, 0), L - 1);

    float fx = x - x0, fy = y - y0, fz = z - z0;

    // Interpolate and apply affine transform in one loop
#pragma unroll
    for (int ci = 0; ci < 12; ci++) {
        // Base pointer for this volume
        int base = ci * L * H * W;

        // Fetch 8 corners with read-only cache hints (shared across threads)
        float v000 = load_ro(&grid[base + (z0 * H + y0) * W + x0]);
        float v001 = load_ro(&grid[base + (z0 * H + y0) * W + x1]);
        float v010 = load_ro(&grid[base + (z0 * H + y1) * W + x0]);
        float v011 = load_ro(&grid[base + (z0 * H + y1) * W + x1]);
        float v100 = load_ro(&grid[base + (z1 * H + y0) * W + x0]);
        float v101 = load_ro(&grid[base + (z1 * H + y0) * W + x1]);
        float v110 = load_ro(&grid[base + (z1 * H + y1) * W + x0]);
        float v111 = load_ro(&grid[base + (z1 * H + y1) * W + x1]);

        // Trilinear interpolation
        float c00 = v000 * (1.0f - fx) + v001 * fx;
        float c01 = v010 * (1.0f - fx) + v011 * fx;
        float c10 = v100 * (1.0f - fx) + v101 * fx;
        float c11 = v110 * (1.0f - fx) + v111 * fx;
        float c0 = c00 * (1.0f - fy) + c01 * fy;
        float c1 = c10 * (1.0f - fy) + c11 * fy;
        float val = c0 * (1.0f - fz) + c1 * fz;

        // Affine transform
        int si = ci % 4; // source index
        int di = ci / 4; // destination index
        (di == 0 ? dr : di == 1 ? dg : db) += val *
            (si == 0 ? sr : si == 1 ? sg : si == 2 ? sb : 1.0f);
    }

    // Write output with safety checks
    output[rgb_idx + 0] = isfinite(dr) ? dr : 0.5f;
    output[rgb_idx + 1] = isfinite(dg) ? dg : 0.5f;
    output[rgb_idx + 2] = isfinite(db) ? db : 0.5f;
}

void launch_bilateral_grid_slice_forward(
    const float* grid,
    const float* rgb,
    float* output,
    int L, int H, int W,
    int h, int w,
    cudaStream_t stream) {

    const int threads = 256;
    const int blocks = (h * w + threads - 1) / threads;

    bilateral_grid_slice_forward_kernel<<<blocks, threads, 0, stream>>>(
        grid, rgb, output, L, H, W, h, w);
}

} // namespace lfs::training::kernels
