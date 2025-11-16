/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_ssim_kernel_wrapper.cu
 * @brief CUDA wrapper to call SSIM kernel from C++ test code
 */

#include <cuda_runtime.h>

// Forward declare the new SSIM kernel
__global__ void fusedssimCUDA(
    int H, int W, int C, float C1, float C2,
    const float* img1, const float* img2,
    float* ssim_map, float* dm_dmu1, float* dm_dsigma1_sq, float* dm_dsigma12);

namespace test_helpers {
    void call_new_ssim_kernel(
        int N, int C, int H, int W, float C1, float C2,
        const float* img1, const float* img2,
        float* ssim_map, float* dm_dmu1, float* dm_dsigma1_sq, float* dm_dsigma12) {

        dim3 grid((W + 16 - 1) / 16, (H + 16 - 1) / 16, N);
        dim3 block(16, 16);

        fusedssimCUDA<<<grid, block>>>(
            H, W, C, C1, C2,
            img1, img2,
            ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);

        cudaDeviceSynchronize();
    }
}
