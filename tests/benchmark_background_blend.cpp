/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "core_new/tensor.hpp"
#include "training_new/kernels/grad_alpha.hpp"
#include <chrono>

using namespace lfs::core;

// Benchmark: Background blending performance
class BackgroundBlendBenchmark : public ::testing::Test {
protected:
    static constexpr int H = 1080;
    static constexpr int W = 1920;
    static constexpr int WARMUP = 10;
    static constexpr int ITERS = 100;

    void SetUp() override {
        // Create test data - use SAME data for LFS and LibTorch for correctness checks
        bg_color_lfs = Tensor::from_vector({0.5f, 0.3f, 0.7f}, {3}, Device::CUDA);
        bg_color_torch = torch::tensor({0.5f, 0.3f, 0.7f}, torch::TensorOptions().device(torch::kCUDA));

        // Create LibTorch tensors, then copy to LFS for exact match
        image_torch = torch::randn({3, H, W}, torch::TensorOptions().device(torch::kCUDA));
        image_lfs = Tensor::empty({3, H, W}, Device::CUDA);
        cudaMemcpy(image_lfs.ptr<float>(), image_torch.data_ptr<float>(),
                   3 * H * W * sizeof(float), cudaMemcpyDeviceToDevice);

        alpha_torch = torch::rand({1, H, W}, torch::TensorOptions().device(torch::kCUDA));
        alpha_lfs = Tensor::empty({1, H, W}, Device::CUDA);
        cudaMemcpy(alpha_lfs.ptr<float>(), alpha_torch.data_ptr<float>(),
                   H * W * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    Tensor bg_color_lfs;
    torch::Tensor bg_color_torch;
    Tensor image_lfs, alpha_lfs;
    torch::Tensor image_torch, alpha_torch;
};

// ==================== LFS Implementations ====================

// OLD: Using separate tensor ops (slow)
Tensor compute_blend_lfs_old(const Tensor& image, const Tensor& alpha, const Tensor& bg_color) {
    auto alpha_complement = (alpha * -1.0f) + 1.0f;  // 1 - alpha
    auto bg_contribution = alpha_complement * bg_color.reshape({3, 1, 1});
    return image + bg_contribution;
}

// NEW: Using fused kernel (fast!)
Tensor compute_blend_lfs_fused(const Tensor& image, const Tensor& alpha, const Tensor& bg_color) {
    int H = static_cast<int>(image.shape()[1]);
    int W = static_cast<int>(image.shape()[2]);

    auto output = Tensor::empty({3, static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);

    lfs::training::kernels::launch_fused_background_blend(
        image.ptr<float>(),
        alpha.ptr<float>(),
        bg_color.ptr<float>(),
        output.ptr<float>(),
        H, W,
        nullptr
    );

    return output;
}

// ==================== LibTorch Implementation ====================

torch::Tensor compute_blend_torch(const torch::Tensor& image, const torch::Tensor& alpha, const torch::Tensor& bg_color) {
    auto alpha_complement = 1.0f - alpha;
    auto bg_contribution = alpha_complement * bg_color.reshape({3, 1, 1});
    return image + bg_contribution;
}

// ==================== Benchmarks ====================

TEST_F(BackgroundBlendBenchmark, LFS_Old) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto result = compute_blend_lfs_old(image_lfs, alpha_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto result = compute_blend_lfs_old(image_lfs, alpha_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERS;

    std::cout << "[LFS OLD] Average time: " << avg_ms << " ms/iter\n";
    std::cout << "[LFS OLD] Total time: " << elapsed_ms << " ms for " << ITERS << " iterations\n";
}

TEST_F(BackgroundBlendBenchmark, LFS_Fused) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto result = compute_blend_lfs_fused(image_lfs, alpha_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto result = compute_blend_lfs_fused(image_lfs, alpha_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERS;

    std::cout << "[LFS FUSED] Average time: " << avg_ms << " ms/iter\n";
    std::cout << "[LFS FUSED] Total time: " << elapsed_ms << " ms for " << ITERS << " iterations\n";
}

TEST_F(BackgroundBlendBenchmark, LibTorch) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto result = compute_blend_torch(image_torch, alpha_torch, bg_color_torch);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto result = compute_blend_torch(image_torch, alpha_torch, bg_color_torch);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERS;

    std::cout << "[LibTorch] Average time: " << avg_ms << " ms/iter\n";
    std::cout << "[LibTorch] Total time: " << elapsed_ms << " ms for " << ITERS << " iterations\n";
}

// ==================== Correctness Test ====================

TEST_F(BackgroundBlendBenchmark, Correctness) {
    auto result_lfs = compute_blend_lfs_fused(image_lfs, alpha_lfs, bg_color_lfs);
    auto result_torch = compute_blend_torch(image_torch, alpha_torch, bg_color_torch);

    // Convert LFS result to CPU for comparison
    auto result_lfs_cpu = result_lfs.to(Device::CPU);
    auto result_torch_cpu = result_torch.cpu();

    // Compare shapes
    ASSERT_EQ(result_lfs.shape().rank(), 3);
    ASSERT_EQ(result_lfs.shape()[0], 3);
    ASSERT_EQ(result_lfs.shape()[1], H);
    ASSERT_EQ(result_lfs.shape()[2], W);

    // Compare values (sample check)
    float* lfs_data = result_lfs_cpu.ptr<float>();
    float* torch_data = result_torch_cpu.data_ptr<float>();

    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < std::min(1000, 3 * H * W); ++i) {
        float diff = std::abs(lfs_data[i] - torch_data[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-4f) {
            mismatches++;
        }
    }

    std::cout << "[Correctness] Max diff: " << max_diff << ", Mismatches: " << mismatches << "/1000\n";
    EXPECT_LT(max_diff, 1e-3f) << "Results differ significantly from LibTorch";
}
