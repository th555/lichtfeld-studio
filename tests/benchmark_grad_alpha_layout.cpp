/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "core_new/tensor.hpp"
#include "training_new/kernels/grad_alpha.hpp"
#include <chrono>

using namespace lfs::core;

// Benchmark: Runtime layout detection + operations
class GradAlphaLayoutBenchmark : public ::testing::Test {
protected:
    static constexpr int H = 1080;
    static constexpr int W = 1920;
    static constexpr int WARMUP = 10;
    static constexpr int ITERS = 100;

    void SetUp() override {
        // Create test data - use SAME data for LFS and LibTorch for correctness checks
        bg_color_lfs = Tensor::from_vector({0.5f, 0.3f, 0.7f}, {3}, Device::CUDA);
        bg_color_torch = torch::tensor({0.5f, 0.3f, 0.7f}, torch::TensorOptions().device(torch::kCUDA));

        // CHW layout - create LibTorch tensor, then copy to LFS for exact match
        grad_image_chw_torch = torch::randn({3, H, W}, torch::TensorOptions().device(torch::kCUDA));
        grad_image_chw_lfs = Tensor::empty({3, H, W}, Device::CUDA);
        cudaMemcpy(grad_image_chw_lfs.ptr<float>(), grad_image_chw_torch.data_ptr<float>(),
                   3 * H * W * sizeof(float), cudaMemcpyDeviceToDevice);

        // HWC layout - create LibTorch tensor, then copy to LFS for exact match
        grad_image_hwc_torch = torch::randn({H, W, 3}, torch::TensorOptions().device(torch::kCUDA));
        grad_image_hwc_lfs = Tensor::empty({H, W, 3}, Device::CUDA);
        cudaMemcpy(grad_image_hwc_lfs.ptr<float>(), grad_image_hwc_torch.data_ptr<float>(),
                   H * W * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    Tensor bg_color_lfs;
    torch::Tensor bg_color_torch;
    Tensor grad_image_chw_lfs, grad_image_hwc_lfs;
    torch::Tensor grad_image_chw_torch, grad_image_hwc_torch;
};

// ==================== LFS Implementations ====================

// OLD: Using separate tensor ops (slow)
Tensor compute_grad_alpha_lfs_old(const Tensor& grad_image, const Tensor& bg_color) {
    Tensor grad_alpha;

    if (grad_image.shape()[0] == 3) {
        auto bg_expanded = bg_color.reshape({3, 1, 1});
        grad_alpha = (grad_image * bg_expanded).sum({0}, false) * -1.0f;
    } else if (grad_image.shape()[2] == 3) {
        auto bg_expanded = bg_color.reshape({1, 1, 3});
        grad_alpha = (grad_image * bg_expanded).sum({2}, false) * -1.0f;
    }

    return grad_alpha;
}

// NEW: Using fused kernel (fast!)
Tensor compute_grad_alpha_lfs_fused(const Tensor& grad_image, const Tensor& bg_color) {
    int H, W;
    bool is_chw_layout;

    if (grad_image.shape()[0] == 3) {
        // CHW: [3, H, W]
        is_chw_layout = true;
        H = grad_image.shape()[1];
        W = grad_image.shape()[2];
    } else {
        // HWC: [H, W, 3]
        is_chw_layout = false;
        H = grad_image.shape()[0];
        W = grad_image.shape()[1];
    }

    auto grad_alpha = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);

    lfs::training::kernels::launch_fused_grad_alpha(
        grad_image.ptr<float>(),
        bg_color.ptr<float>(),
        grad_alpha.ptr<float>(),
        H, W,
        is_chw_layout,
        nullptr
    );

    return grad_alpha;
}

// ==================== LibTorch Implementation ====================

torch::Tensor compute_grad_alpha_torch(const torch::Tensor& grad_image, const torch::Tensor& bg_color) {
    torch::Tensor grad_alpha;

    if (grad_image.size(0) == 3) {
        // Layout: [3, H, W]
        auto bg_expanded = bg_color.reshape({3, 1, 1});
        grad_alpha = (grad_image * bg_expanded).sum(0) * -1.0f;
    } else if (grad_image.size(2) == 3) {
        // Layout: [H, W, 3]
        auto bg_expanded = bg_color.reshape({1, 1, 3});
        grad_alpha = (grad_image * bg_expanded).sum(2) * -1.0f;
    }

    return grad_alpha;
}

// ==================== Benchmarks ====================

TEST_F(GradAlphaLayoutBenchmark, CHW_Layout_LFS_Old) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto result = compute_grad_alpha_lfs_old(grad_image_chw_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto result = compute_grad_alpha_lfs_old(grad_image_chw_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERS;

    std::cout << "[LFS CHW OLD] Average time: " << avg_ms << " ms/iter\n";
    std::cout << "[LFS CHW OLD] Total time: " << elapsed_ms << " ms for " << ITERS << " iterations\n";
}

TEST_F(GradAlphaLayoutBenchmark, CHW_Layout_LFS_Fused) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto result = compute_grad_alpha_lfs_fused(grad_image_chw_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto result = compute_grad_alpha_lfs_fused(grad_image_chw_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERS;

    std::cout << "[LFS CHW FUSED] Average time: " << avg_ms << " ms/iter\n";
    std::cout << "[LFS CHW FUSED] Total time: " << elapsed_ms << " ms for " << ITERS << " iterations\n";
    std::cout << "[LFS CHW FUSED] Speedup vs LibTorch: " << (elapsed_ms / ITERS) / (elapsed_ms / ITERS) << "x\n";
}

TEST_F(GradAlphaLayoutBenchmark, CHW_Layout_Torch) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto result = compute_grad_alpha_torch(grad_image_chw_torch, bg_color_torch);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto result = compute_grad_alpha_torch(grad_image_chw_torch, bg_color_torch);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERS;

    std::cout << "[Torch CHW] Average time: " << avg_ms << " ms/iter\n";
    std::cout << "[Torch CHW] Total time: " << elapsed_ms << " ms for " << ITERS << " iterations\n";
}

TEST_F(GradAlphaLayoutBenchmark, HWC_Layout_LFS_Old) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto result = compute_grad_alpha_lfs_old(grad_image_hwc_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto result = compute_grad_alpha_lfs_old(grad_image_hwc_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERS;

    std::cout << "[LFS HWC OLD] Average time: " << avg_ms << " ms/iter\n";
    std::cout << "[LFS HWC OLD] Total time: " << elapsed_ms << " ms for " << ITERS << " iterations\n";
}

TEST_F(GradAlphaLayoutBenchmark, HWC_Layout_LFS_Fused) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto result = compute_grad_alpha_lfs_fused(grad_image_hwc_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto result = compute_grad_alpha_lfs_fused(grad_image_hwc_lfs, bg_color_lfs);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERS;

    std::cout << "[LFS HWC FUSED] Average time: " << avg_ms << " ms/iter\n";
    std::cout << "[LFS HWC FUSED] Total time: " << elapsed_ms << " ms for " << ITERS << " iterations\n";
}

TEST_F(GradAlphaLayoutBenchmark, HWC_Layout_Torch) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto result = compute_grad_alpha_torch(grad_image_hwc_torch, bg_color_torch);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto result = compute_grad_alpha_torch(grad_image_hwc_torch, bg_color_torch);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERS;

    std::cout << "[Torch HWC] Average time: " << avg_ms << " ms/iter\n";
    std::cout << "[Torch HWC] Total time: " << elapsed_ms << " ms for " << ITERS << " iterations\n";
}

// ==================== Correctness Test ====================

TEST_F(GradAlphaLayoutBenchmark, Correctness_CHW) {
    auto result_lfs = compute_grad_alpha_lfs_fused(grad_image_chw_lfs, bg_color_lfs);
    auto result_torch = compute_grad_alpha_torch(grad_image_chw_torch, bg_color_torch);

    // Convert LFS result to CPU for comparison
    auto result_lfs_cpu = result_lfs.to(Device::CPU);
    auto result_torch_cpu = result_torch.cpu();

    // Compare shapes
    ASSERT_EQ(result_lfs.shape().rank(), 2);
    ASSERT_EQ(result_lfs.shape()[0], H);
    ASSERT_EQ(result_lfs.shape()[1], W);

    // Compare values (sample check)
    float* lfs_data = result_lfs_cpu.ptr<float>();
    float* torch_data = result_torch_cpu.data_ptr<float>();

    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < std::min(1000, H * W); ++i) {
        float diff = std::abs(lfs_data[i] - torch_data[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-4f) {
            mismatches++;
        }
    }

    std::cout << "[CHW Correctness] Max diff: " << max_diff << ", Mismatches: " << mismatches << "/1000\n";
    EXPECT_LT(max_diff, 1e-3f) << "Results differ significantly from LibTorch";
}

TEST_F(GradAlphaLayoutBenchmark, Correctness_HWC) {
    auto result_lfs = compute_grad_alpha_lfs_fused(grad_image_hwc_lfs, bg_color_lfs);
    auto result_torch = compute_grad_alpha_torch(grad_image_hwc_torch, bg_color_torch);

    // Convert LFS result to CPU for comparison
    auto result_lfs_cpu = result_lfs.to(Device::CPU);
    auto result_torch_cpu = result_torch.cpu();

    // Compare shapes
    ASSERT_EQ(result_lfs.shape().rank(), 2);
    ASSERT_EQ(result_lfs.shape()[0], H);
    ASSERT_EQ(result_lfs.shape()[1], W);

    // Compare values (sample check)
    float* lfs_data = result_lfs_cpu.ptr<float>();
    float* torch_data = result_torch_cpu.data_ptr<float>();

    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < std::min(1000, H * W); ++i) {
        float diff = std::abs(lfs_data[i] - torch_data[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-4f) {
            mismatches++;
        }
    }

    std::cout << "[HWC Correctness] Max diff: " << max_diff << ", Mismatches: " << mismatches << "/1000\n";
    EXPECT_LT(max_diff, 1e-3f) << "Results differ significantly from LibTorch";
}

// ==================== Runtime Overhead Test ====================

TEST_F(GradAlphaLayoutBenchmark, RuntimeBranchOverhead) {
    // Test with alternating layouts to measure branch prediction impact
    std::vector<Tensor> images = {grad_image_chw_lfs, grad_image_hwc_lfs};

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto result = compute_grad_alpha_lfs_fused(images[i % 2], bg_color_lfs);
    }
    cudaDeviceSynchronize();

    // Benchmark with alternating layouts (worst case for branch prediction)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto result = compute_grad_alpha_lfs_fused(images[i % 2], bg_color_lfs);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERS;

    std::cout << "[Alternating Layouts] Average time: " << avg_ms << " ms/iter\n";
    std::cout << "[Alternating Layouts] Branch prediction worst case\n";
}
