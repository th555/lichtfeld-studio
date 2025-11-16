/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "config.h"

#ifdef CUDA_GL_INTEROP_ENABLED

#include <cuda_runtime.h>

namespace lfs {

    // CUDA kernel to interleave position and color data
    __global__ void writeInterleavedPosColorKernel(
        const float* __restrict__ positions, // [N, 3]
        const float* __restrict__ colors,    // [N, 3]
        float* __restrict__ output,          // [N, 6] interleaved
        int num_points) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points)
            return;

        // Read position (x, y, z)
        float px = positions[idx * 3 + 0];
        float py = positions[idx * 3 + 1];
        float pz = positions[idx * 3 + 2];

        // Read color (r, g, b)
        float cr = colors[idx * 3 + 0];
        float cg = colors[idx * 3 + 1];
        float cb = colors[idx * 3 + 2];

        // Write interleaved (x, y, z, r, g, b)
        int out_idx = idx * 6;
        output[out_idx + 0] = px;
        output[out_idx + 1] = py;
        output[out_idx + 2] = pz;
        output[out_idx + 3] = cr;
        output[out_idx + 4] = cg;
        output[out_idx + 5] = cb;
    }

    // Host function to launch the kernel
    void launchWriteInterleavedPosColor(
        const float* positions,
        const float* colors,
        float* output,
        int num_points,
        cudaStream_t stream) {

        if (num_points <= 0)
            return;

        const int threads = 256;
        const int blocks = (num_points + threads - 1) / threads;

        writeInterleavedPosColorKernel<<<blocks, threads, 0, stream>>>(
            positions, colors, output, num_points);
    }

} // namespace lfs

#endif // CUDA_GL_INTEROP_ENABLED
