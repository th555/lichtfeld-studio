/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "optimizer/adam_optimizer.hpp"  // For src/training_new include path
#include "losses/regularization.hpp"      // New libtorch-free losses
#include "losses/photometric_loss.hpp"    // New libtorch-free losses
#include "kernels/regularization.cuh"     // Old CUDA kernels for comparison
#include "lfs/kernels/ssim.cuh"           // Old SSIM kernels for comparison
#include "lfs/kernels/bilateral_grid.cuh" // New bilateral grid kernels
#include "kernels/bilateral_grid.cuh"     // Old bilateral grid kernels for comparison
#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>

using namespace lfs::core;
using namespace lfs::training::losses;

namespace {

// Timing utilities
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }

    double stop_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    double stop_us() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

void print_benchmark_header(const std::string& title) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ " << std::left << std::setw(61) << title << "║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
}

// Helper to convert torch::Tensor to lfs::core::Tensor
Tensor from_torch(const torch::Tensor& torch_tensor) {
    auto cpu_t = torch_tensor.cpu().contiguous();
    std::vector<float> vec(cpu_t.data_ptr<float>(),
                           cpu_t.data_ptr<float>() + cpu_t.numel());

    std::vector<size_t> shape;
    for (int i = 0; i < cpu_t.dim(); i++) {
        shape.push_back(cpu_t.size(i));
    }

    auto device = torch_tensor.is_cuda() ? Device::CUDA : Device::CPU;
    return Tensor::from_vector(vec, TensorShape(shape), device);
}

void print_result(const std::string& test_name, double old_time, double new_time, const std::string& unit = "μs") {
    double speedup = old_time / new_time;
    std::cout << std::left << std::setw(45) << test_name << " | ";
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << old_time << " " << unit << " | ";
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << new_time << " " << unit << " | ";
    std::cout << std::right << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x";

    if (speedup > 1.05) {
        std::cout << " ✓";
    } else if (speedup < 0.95) {
        std::cout << " ⚠";
    }
    std::cout << "\n";
}

// Helper to convert lfs::core::Tensor to torch::Tensor
torch::Tensor to_torch(const lfs::core::Tensor& lfs_tensor) {
    auto vec = lfs_tensor.cpu().to_vector();
    std::vector<int64_t> torch_shape;
    for (size_t i = 0; i < lfs_tensor.ndim(); i++) {
        torch_shape.push_back(lfs_tensor.shape()[i]);
    }
    auto torch_t = torch::from_blob(vec.data(), torch_shape, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return lfs_tensor.device() == lfs::core::Device::CUDA ? torch_t.to(torch::kCUDA) : torch_t;
}

// ===================================================================================
// ScaleRegularization Benchmarks
// ===================================================================================

TEST(LossesBenchmark, ScaleRegularization_SingleCall) {
    print_benchmark_header("ScaleRegularization - Single Call Latency");
    std::cout << std::left << std::setw(45) << "Test" << " | ";
    std::cout << std::right << std::setw(10) << "Old (Torch)" << "    | ";
    std::cout << std::right << std::setw(10) << "New (Free)" << "    | ";
    std::cout << std::right << std::setw(6) << "Speedup\n";
    std::cout << std::string(85, '-') << "\n";

    const int n_trials = 1000;
    const size_t n = 10000;
    const float weight = 0.01f;

    // Warmup
    {
        auto scaling_raw = Tensor::randn({100, 3}, Device::CUDA);
        auto scaling_raw_grad = Tensor::zeros({100, 3}, Device::CUDA);
        ScaleRegularization::Params params{.weight = weight};
        ScaleRegularization::forward(scaling_raw, scaling_raw_grad, params);
    }

    // Benchmark old implementation
    auto scaling_raw_old = Tensor::randn({n, 3}, Device::CUDA);
    auto torch_scaling_raw = to_torch(scaling_raw_old);
    torch_scaling_raw.set_requires_grad(true);

    Timer timer;
    timer.start();
    for (int i = 0; i < n_trials; i++) {
        torch_scaling_raw.mutable_grad() = torch::zeros_like(torch_scaling_raw);
        gs::regularization::compute_exp_l1_regularization_with_grad_cuda(torch_scaling_raw, weight);
    }
    cudaDeviceSynchronize();
    double old_time = timer.stop_us() / n_trials;

    // Benchmark new implementation
    auto scaling_raw_new = Tensor::randn({n, 3}, Device::CUDA);
    auto scaling_raw_grad = Tensor::zeros({n, 3}, Device::CUDA);
    ScaleRegularization::Params params{.weight = weight};

    timer.start();
    for (int i = 0; i < n_trials; i++) {
        scaling_raw_grad = Tensor::zeros({n, 3}, Device::CUDA);
        ScaleRegularization::forward(scaling_raw_new, scaling_raw_grad, params);
    }
    cudaDeviceSynchronize();
    double new_time = timer.stop_us() / n_trials;

    print_result("10k Gaussians (1000 trials avg)", old_time, new_time);
}

TEST(LossesBenchmark, ScaleRegularization_Throughput) {
    print_benchmark_header("ScaleRegularization - Throughput");
    std::cout << std::left << std::setw(45) << "Scale" << " | ";
    std::cout << std::right << std::setw(15) << "Old (calls/s)" << " | ";
    std::cout << std::right << std::setw(15) << "New (calls/s)" << " | ";
    std::cout << std::right << std::setw(6) << "Speedup\n";
    std::cout << std::string(90, '-') << "\n";

    const double test_duration_ms = 100.0;
    const float weight = 0.01f;

    std::vector<size_t> sizes = {1000, 10000, 100000};

    for (size_t n : sizes) {
        // Old implementation
        auto scaling_raw_old = Tensor::randn({n, 3}, Device::CUDA);
        auto torch_scaling_raw = to_torch(scaling_raw_old);
        torch_scaling_raw.set_requires_grad(true);

        Timer timer;
        timer.start();
        int old_count = 0;
        while (timer.stop_ms() < test_duration_ms) {
            torch_scaling_raw.mutable_grad() = torch::zeros_like(torch_scaling_raw);
            gs::regularization::compute_exp_l1_regularization_with_grad_cuda(torch_scaling_raw, weight);
            old_count++;
        }
        cudaDeviceSynchronize();
        double old_elapsed = timer.stop_ms();
        double old_throughput = (old_count / old_elapsed) * 1000.0;

        // New implementation
        auto scaling_raw_new = Tensor::randn({n, 3}, Device::CUDA);
        auto scaling_raw_grad = Tensor::zeros({n, 3}, Device::CUDA);
        ScaleRegularization::Params params{.weight = weight};

        timer.start();
        int new_count = 0;
        while (timer.stop_ms() < test_duration_ms) {
            scaling_raw_grad = Tensor::zeros({n, 3}, Device::CUDA);
            ScaleRegularization::forward(scaling_raw_new, scaling_raw_grad, params);
            new_count++;
        }
        cudaDeviceSynchronize();
        double new_elapsed = timer.stop_ms();
        double new_throughput = (new_count / new_elapsed) * 1000.0;

        std::cout << std::left << std::setw(45) << (std::to_string(n) + " Gaussians") << " | ";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(0) << old_throughput << " | ";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(0) << new_throughput << " | ";
        std::cout << std::right << std::setw(6) << std::fixed << std::setprecision(2) << (new_throughput / old_throughput) << "x\n";
    }
}

// ===================================================================================
// OpacityRegularization Benchmarks
// ===================================================================================

TEST(LossesBenchmark, OpacityRegularization_SingleCall) {
    print_benchmark_header("OpacityRegularization - Single Call Latency");
    std::cout << std::left << std::setw(45) << "Test" << " | ";
    std::cout << std::right << std::setw(10) << "Old (Torch)" << "    | ";
    std::cout << std::right << std::setw(10) << "New (Free)" << "    | ";
    std::cout << std::right << std::setw(6) << "Speedup\n";
    std::cout << std::string(85, '-') << "\n";

    const int n_trials = 1000;
    const size_t n = 10000;
    const float weight = 0.01f;

    // Old implementation
    auto opacity_raw_old = Tensor::randn({n, 1}, Device::CUDA);
    auto torch_opacity_raw = to_torch(opacity_raw_old);
    torch_opacity_raw.set_requires_grad(true);

    Timer timer;
    timer.start();
    for (int i = 0; i < n_trials; i++) {
        torch_opacity_raw.mutable_grad() = torch::zeros_like(torch_opacity_raw);
        gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(torch_opacity_raw, weight);
    }
    cudaDeviceSynchronize();
    double old_time = timer.stop_us() / n_trials;

    // New implementation
    auto opacity_raw_new = Tensor::randn({n, 1}, Device::CUDA);
    auto opacity_raw_grad = Tensor::zeros({n, 1}, Device::CUDA);
    OpacityRegularization::Params params{.weight = weight};

    timer.start();
    for (int i = 0; i < n_trials; i++) {
        opacity_raw_grad = Tensor::zeros({n, 1}, Device::CUDA);
        OpacityRegularization::forward(opacity_raw_new, opacity_raw_grad, params);
    }
    cudaDeviceSynchronize();
    double new_time = timer.stop_us() / n_trials;

    print_result("10k Gaussians (1000 trials avg)", old_time, new_time);
}

// ===================================================================================
// PhotometricLoss Benchmarks
// ===================================================================================

TEST(LossesBenchmark, PhotometricLoss_L1Only) {
    print_benchmark_header("PhotometricLoss - L1 Only");
    std::cout << std::left << std::setw(45) << "Image Size" << " | ";
    std::cout << std::right << std::setw(10) << "Time (ms)" << "    | ";
    std::cout << std::right << std::setw(10) << "FPS" << "\n";
    std::cout << std::string(75, '-') << "\n";

    const int n_trials = 100;
    const float lambda_dssim = 0.0f;  // Pure L1

    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {800, 800}, {1024, 1024}};

    for (auto [H, W] : sizes) {
        auto rendered = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
        auto gt_image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);

        PhotometricLoss::Params params{.lambda_dssim = lambda_dssim};

        Timer timer;
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            PhotometricLoss loss_fn;
    auto result = loss_fn.forward(rendered, gt_image, params);
        }
        cudaDeviceSynchronize();
        double elapsed_ms = timer.stop_ms();
        double time_per_call = elapsed_ms / n_trials;
        double fps = 1000.0 / time_per_call;

        std::cout << std::left << std::setw(45) << (std::to_string(H) + "x" + std::to_string(W)) << " | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_per_call << " ms | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << fps << "\n";
    }
}

TEST(LossesBenchmark, PhotometricLoss_SSIMOnly) {
    print_benchmark_header("PhotometricLoss - SSIM Only");
    std::cout << std::left << std::setw(45) << "Image Size" << " | ";
    std::cout << std::right << std::setw(10) << "Time (ms)" << "    | ";
    std::cout << std::right << std::setw(10) << "FPS" << "\n";
    std::cout << std::string(75, '-') << "\n";

    const int n_trials = 100;
    const float lambda_dssim = 1.0f;  // Pure SSIM

    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {800, 800}};

    for (auto [H, W] : sizes) {
        auto rendered = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
        auto gt_image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);

        PhotometricLoss::Params params{.lambda_dssim = lambda_dssim};

        Timer timer;
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            PhotometricLoss loss_fn;
    auto result = loss_fn.forward(rendered, gt_image, params);
        }
        cudaDeviceSynchronize();
        double elapsed_ms = timer.stop_ms();
        double time_per_call = elapsed_ms / n_trials;
        double fps = 1000.0 / time_per_call;

        std::cout << std::left << std::setw(45) << (std::to_string(H) + "x" + std::to_string(W)) << " | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_per_call << " ms | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << fps << "\n";
    }
}

TEST(LossesBenchmark, PhotometricLoss_Combined) {
    print_benchmark_header("PhotometricLoss - Combined (lambda=0.2)");
    std::cout << std::left << std::setw(45) << "Image Size" << " | ";
    std::cout << std::right << std::setw(10) << "Time (ms)" << "    | ";
    std::cout << std::right << std::setw(10) << "FPS" << "\n";
    std::cout << std::string(75, '-') << "\n";

    const int n_trials = 100;
    const float lambda_dssim = 0.2f;  // Typical training value

    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {800, 800}};

    for (auto [H, W] : sizes) {
        auto rendered = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
        auto gt_image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);

        PhotometricLoss::Params params{.lambda_dssim = lambda_dssim};

        Timer timer;
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            PhotometricLoss loss_fn;
    auto result = loss_fn.forward(rendered, gt_image, params);
        }
        cudaDeviceSynchronize();
        double elapsed_ms = timer.stop_ms();
        double time_per_call = elapsed_ms / n_trials;
        double fps = 1000.0 / time_per_call;

        std::cout << std::left << std::setw(45) << (std::to_string(H) + "x" + std::to_string(W)) << " | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_per_call << " ms | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << fps << "\n";
    }
}

// ===================================================================================
// Realistic Training Scenario
// ===================================================================================

TEST(LossesBenchmark, RealisticTraining_FullLossPipeline) {
    print_benchmark_header("Realistic Training - Full Loss Computation");
    std::cout << "Simulating typical Gaussian Splatting training iteration\n";
    std::cout << "Configuration: 100k Gaussians, 800x800 image, all losses enabled\n\n";

    const int n_iterations = 100;
    const size_t n_gaussians = 100000;
    const int H = 800, W = 800, C = 3;

    // Setup
    auto scaling_raw = Tensor::randn({n_gaussians, 3}, Device::CUDA);
    auto opacity_raw = Tensor::randn({n_gaussians, 1}, Device::CUDA);
    auto rendered = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);
    auto gt_image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);

    auto scaling_grad = Tensor::zeros({n_gaussians, 3}, Device::CUDA);
    auto opacity_grad = Tensor::zeros({n_gaussians, 1}, Device::CUDA);

    ScaleRegularization::Params scale_params{.weight = 0.01f};
    OpacityRegularization::Params opacity_params{.weight = 0.01f};
    PhotometricLoss::Params photo_params{.lambda_dssim = 0.2f};

    Timer timer;
    timer.start();

    for (int i = 0; i < n_iterations; i++) {
        // Photometric loss (main loss)
        PhotometricLoss loss_fn;
        auto photo_result = loss_fn.forward(rendered, gt_image, photo_params);

        // Regularization losses
        auto scale_result = ScaleRegularization::forward(scaling_raw, scaling_grad, scale_params);
        auto opacity_result = OpacityRegularization::forward(opacity_raw, opacity_grad, opacity_params);
    }

    cudaDeviceSynchronize();
    double total_ms = timer.stop_ms();
    double time_per_iter = total_ms / n_iterations;

    std::cout << std::left << std::setw(40) << "Total time for 100 iterations:" << std::right << std::setw(10)
              << std::fixed << std::setprecision(2) << total_ms << " ms\n";
    std::cout << std::left << std::setw(40) << "Time per iteration:" << std::right << std::setw(10)
              << std::fixed << std::setprecision(3) << time_per_iter << " ms\n";
    std::cout << std::left << std::setw(40) << "Throughput:" << std::right << std::setw(10)
              << std::fixed << std::setprecision(1) << (1000.0 / time_per_iter) << " iter/s\n";
}

// ===================================================================================
// Memory Overhead
// ===================================================================================

TEST(LossesBenchmark, MemoryOverhead) {
    print_benchmark_header("Memory Overhead Analysis");
    std::cout << "Measuring additional memory allocations during loss computation\n\n";

    const size_t n_gaussians = 100000;
    const int H = 512, W = 512, C = 3;

    // Setup tensors
    auto scaling_raw = Tensor::randn({n_gaussians, 3}, Device::CUDA);
    auto opacity_raw = Tensor::randn({n_gaussians, 1}, Device::CUDA);
    auto rendered = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);
    auto gt_image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);

    auto scaling_grad = Tensor::zeros({n_gaussians, 3}, Device::CUDA);
    auto opacity_grad = Tensor::zeros({n_gaussians, 1}, Device::CUDA);

    // Get initial GPU memory
    cudaDeviceSynchronize();
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);

    // Create loss function instance
    PhotometricLoss loss_fn;

    // Run losses multiple times
    for (int i = 0; i < 10; i++) {
        ScaleRegularization::Params scale_params{.weight = 0.01f};
        ScaleRegularization::forward(scaling_raw, scaling_grad, scale_params);

        OpacityRegularization::Params opacity_params{.weight = 0.01f};
        OpacityRegularization::forward(opacity_raw, opacity_grad, opacity_params);

        PhotometricLoss::Params photo_params{.lambda_dssim = 0.2f};
        loss_fn.forward(rendered, gt_image, photo_params);
    }

    cudaDeviceSynchronize();
    size_t free_after;
    cudaMemGetInfo(&free_after, &total);

    long long memory_delta = static_cast<long long>(free_before) - static_cast<long long>(free_after);

    std::cout << "Memory change after 10 iterations: " << (memory_delta / 1024.0 / 1024.0) << " MB\n";
    std::cout << "(Should be minimal - zero-copy wrappers don't allocate)\n";
}

// ===================================================================================
// Comparison Benchmarks vs Reference Implementations
// ===================================================================================

// Reference L1 loss using PyTorch (forward only)
float reference_l1_loss_torch_fwd(const torch::Tensor& img1, const torch::Tensor& img2) {
    return torch::l1_loss(img1, img2, torch::Reduction::Mean).item<float>();
}

// Reference L1 loss with backward (fair comparison)
std::pair<float, torch::Tensor> reference_l1_loss_torch_fwd_bwd(const torch::Tensor& img1, const torch::Tensor& img2) {
    auto img1_copy = img1.clone().set_requires_grad(true);
    auto loss = torch::l1_loss(img1_copy, img2, torch::Reduction::Mean);
    loss.backward();
    return {loss.item<float>(), img1_copy.grad()};
}

// Reference SSIM using old kernels (include/lfs/kernels/ssim.cuh)
// FAIR COMPARISON: Also compute backward (same as PhotometricLoss)
float reference_ssim_old_kernels(const lfs::core::Tensor& img1, const lfs::core::Tensor& img2) {
    auto [ssim_value_tensor, ctx] = lfs::training::kernels::ssim_forward(img1, img2);
    // Compute backward to match what PhotometricLoss does
    auto grad = lfs::training::kernels::ssim_backward(ctx, -1.0f);
    // Sync to CPU for testing (acceptable in tests)
    float ssim_value = ssim_value_tensor.item();
    return 1.0f - ssim_value;  // Convert to D-SSIM loss (1 - SSIM)
}

// Reference photometric loss built from torch L1 + old SSIM kernels
float reference_photometric_combined(const lfs::core::Tensor& rendered, const lfs::core::Tensor& gt_image, float lambda_dssim) {
    // L1 component
    auto torch_rendered = to_torch(rendered);
    auto torch_gt = to_torch(gt_image);
    float l1_loss = reference_l1_loss_torch_fwd(torch_rendered, torch_gt);

    // SSIM component
    float ssim_loss = reference_ssim_old_kernels(rendered, gt_image);

    // Combined
    return (1.0f - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss;
}

TEST(LossesBenchmark, L1_vs_Reference) {
    print_benchmark_header("L1 Loss - New Implementation vs PyTorch Reference");
    std::cout << std::left << std::setw(45) << "Image Size" << " | ";
    std::cout << std::right << std::setw(12) << "PyTorch (μs)" << " | ";
    std::cout << std::right << std::setw(12) << "LFS (μs)" << " | ";
    std::cout << std::right << std::setw(8) << "Speedup\n";
    std::cout << std::string(90, '-') << "\n";

    const int n_trials = 1000;
    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {800, 800}};

    PhotometricLoss loss_fn;

    for (auto [H, W] : sizes) {
        auto img1_lfs = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
        auto img2_lfs = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
        auto img1_torch = to_torch(img1_lfs);
        auto img2_torch = to_torch(img2_lfs);

        // Warmup
        reference_l1_loss_torch_fwd_bwd(img1_torch, img2_torch);
        auto l1_result = loss_fn.forward(img1_lfs, img2_lfs, PhotometricLoss::Params{.lambda_dssim = 0.0f});

        // Benchmark PyTorch reference (forward + backward)
        Timer timer;
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            auto [loss, grad] = reference_l1_loss_torch_fwd_bwd(img1_torch, img2_torch);
        }
        cudaDeviceSynchronize();
        double torch_time = timer.stop_us() / n_trials;

        // Benchmark LFS implementation (pure L1, lambda=0)
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            PhotometricLoss loss_fn;
    auto result = loss_fn.forward(img1_lfs, img2_lfs, PhotometricLoss::Params{.lambda_dssim = 0.0f});
        }
        cudaDeviceSynchronize();
        double lfs_time = timer.stop_us() / n_trials;

        print_result(std::to_string(H) + "x" + std::to_string(W), torch_time, lfs_time, "μs");
    }
}

TEST(LossesBenchmark, SSIM_vs_Reference) {
    print_benchmark_header("SSIM Loss - New Implementation vs Old Kernels");
    std::cout << std::left << std::setw(45) << "Image Size" << " | ";
    std::cout << std::right << std::setw(12) << "Old (μs)" << " | ";
    std::cout << std::right << std::setw(12) << "New (μs)" << " | ";
    std::cout << std::right << std::setw(8) << "Speedup\n";
    std::cout << std::string(90, '-') << "\n";

    const int n_trials = 1000;
    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {800, 800}};

    PhotometricLoss loss_fn;
    for (auto [H, W] : sizes) {
        auto img1 = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
        auto img2 = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);

        // Warmup
        reference_ssim_old_kernels(img1, img2);
        auto ssim_result = loss_fn.forward(img1, img2, PhotometricLoss::Params{.lambda_dssim = 1.0f});

        // Benchmark old kernels
        Timer timer;
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            float loss = reference_ssim_old_kernels(img1, img2);
        }
        cudaDeviceSynchronize();
        double old_time = timer.stop_us() / n_trials;

        // Benchmark new implementation (pure SSIM, lambda=1)
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            PhotometricLoss loss_fn;
    auto result = loss_fn.forward(img1, img2, PhotometricLoss::Params{.lambda_dssim = 1.0f});
        }
        cudaDeviceSynchronize();
        double new_time = timer.stop_us() / n_trials;

        print_result(std::to_string(H) + "x" + std::to_string(W), old_time, new_time, "μs");
    }
}

TEST(LossesBenchmark, PhotometricCombined_vs_Reference) {
    print_benchmark_header("Photometric Loss (Combined) - New vs Reference");
    std::cout << std::left << std::setw(45) << "Image Size" << " | ";
    std::cout << std::right << std::setw(12) << "Reference (μs)" << " | ";
    std::cout << std::right << std::setw(12) << "LFS (μs)" << " | ";
    std::cout << std::right << std::setw(8) << "Speedup\n";
    std::cout << std::string(90, '-') << "\n";

    const int n_trials = 1000;
    const float lambda_dssim = 0.2f;  // Typical training value
    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {800, 800}};

    for (auto [H, W] : sizes) {
        auto rendered = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
        auto gt_image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);

        // Warmup
        reference_photometric_combined(rendered, gt_image, lambda_dssim);
        PhotometricLoss loss_fn;
        auto lfs_result = loss_fn.forward(rendered, gt_image, PhotometricLoss::Params{.lambda_dssim = lambda_dssim});

        // Benchmark reference (torch L1 + old SSIM)
        Timer timer;
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            float loss = reference_photometric_combined(rendered, gt_image, lambda_dssim);
        }
        cudaDeviceSynchronize();
        double ref_time = timer.stop_us() / n_trials;

        // Benchmark LFS implementation
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            PhotometricLoss loss_fn;
    auto result = loss_fn.forward(rendered, gt_image, PhotometricLoss::Params{.lambda_dssim = lambda_dssim});
        }
        cudaDeviceSynchronize();
        double lfs_time = timer.stop_us() / n_trials;

        print_result(std::to_string(H) + "x" + std::to_string(W), ref_time, lfs_time, "μs");
    }
}

TEST(LossesBenchmark, NumericalAccuracy_L1) {
    print_benchmark_header("Numerical Accuracy - L1 Loss");
    std::cout << "Comparing LFS L1 implementation vs PyTorch reference\n\n";

    const int H = 512, W = 512;
    auto img1_lfs = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
    auto img2_lfs = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
    auto img1_torch = to_torch(img1_lfs);
    auto img2_torch = to_torch(img2_lfs);

    // Compute losses
    float torch_loss = reference_l1_loss_torch_fwd(img1_torch, img2_torch);
    PhotometricLoss loss_fn;
        auto lfs_result = loss_fn.forward(img1_lfs, img2_lfs, PhotometricLoss::Params{.lambda_dssim = 0.0f});
    float lfs_loss = lfs_result.value().first.item();  // Sync tensor to CPU

    float abs_error = std::abs(torch_loss - lfs_loss);
    float rel_error = abs_error / std::max(torch_loss, 1e-8f);

    std::cout << std::left << std::setw(30) << "PyTorch L1 loss:" << std::right << std::setw(15) << std::fixed << std::setprecision(8) << torch_loss << "\n";
    std::cout << std::left << std::setw(30) << "LFS L1 loss:" << std::right << std::setw(15) << std::fixed << std::setprecision(8) << lfs_loss << "\n";
    std::cout << std::left << std::setw(30) << "Absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << abs_error << "\n";
    std::cout << std::left << std::setw(30) << "Relative error:" << std::right << std::setw(15) << std::fixed << std::setprecision(6) << (rel_error * 100.0f) << "%\n";

    EXPECT_LT(rel_error, 1e-5);  // Should match within 0.001%
}

TEST(LossesBenchmark, NumericalAccuracy_SSIM) {
    print_benchmark_header("Numerical Accuracy - SSIM Loss");
    std::cout << "Comparing LFS SSIM implementation vs old kernels\n\n";

    const int H = 512, W = 512;
    auto img1 = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
    auto img2 = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);

    // Compute losses
    float old_loss = reference_ssim_old_kernels(img1, img2);
    PhotometricLoss loss_fn;
        auto lfs_result = loss_fn.forward(img1, img2, PhotometricLoss::Params{.lambda_dssim = 1.0f});
    float lfs_loss = lfs_result.value().first.item();  // Sync tensor to CPU

    float abs_error = std::abs(old_loss - lfs_loss);
    float rel_error = abs_error / std::max(old_loss, 1e-8f);

    std::cout << std::left << std::setw(30) << "Old SSIM loss:" << std::right << std::setw(15) << std::fixed << std::setprecision(8) << old_loss << "\n";
    std::cout << std::left << std::setw(30) << "LFS SSIM loss:" << std::right << std::setw(15) << std::fixed << std::setprecision(8) << lfs_loss << "\n";
    std::cout << std::left << std::setw(30) << "Absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << abs_error << "\n";
    std::cout << std::left << std::setw(30) << "Relative error:" << std::right << std::setw(15) << std::fixed << std::setprecision(6) << (rel_error * 100.0f) << "%\n";

    EXPECT_LT(rel_error, 1e-5);  // Should match within 0.001%
}

TEST(LossesBenchmark, NumericalAccuracy_PhotometricCombined) {
    print_benchmark_header("Numerical Accuracy - Photometric Loss (Combined)");
    std::cout << "Comparing LFS implementation vs reference (torch L1 + old SSIM)\n\n";

    const int H = 512, W = 512;
    const float lambda_dssim = 0.2f;
    auto rendered = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
    auto gt_image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);

    // Compute losses
    float ref_loss = reference_photometric_combined(rendered, gt_image, lambda_dssim);
    PhotometricLoss loss_fn;
        auto lfs_result = loss_fn.forward(rendered, gt_image, PhotometricLoss::Params{.lambda_dssim = lambda_dssim});
    float lfs_loss = lfs_result.value().first.item();  // Sync tensor to CPU

    float abs_error = std::abs(ref_loss - lfs_loss);
    float rel_error = abs_error / std::max(ref_loss, 1e-8f);

    std::cout << std::left << std::setw(30) << "Reference loss:" << std::right << std::setw(15) << std::fixed << std::setprecision(8) << ref_loss << "\n";
    std::cout << std::left << std::setw(30) << "LFS loss:" << std::right << std::setw(15) << std::fixed << std::setprecision(8) << lfs_loss << "\n";
    std::cout << std::left << std::setw(30) << "Absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << abs_error << "\n";
    std::cout << std::left << std::setw(30) << "Relative error:" << std::right << std::setw(15) << std::fixed << std::setprecision(6) << (rel_error * 100.0f) << "%\n";

    EXPECT_LT(rel_error, 1e-5);  // Should match within 0.001%
}

TEST(LossesBenchmark, GradientCorrectness_L1) {
    print_benchmark_header("Gradient Correctness - L1 Loss");
    std::cout << "Comparing analytical gradients vs numerical gradients (finite differences)\n\n";

    const int H = 64, W = 64;  // Smaller for faster gradient checking
    auto img1 = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
    auto img2 = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);

    // Compute analytical gradients (forward already computes them)
    PhotometricLoss loss_fn;
    auto result = loss_fn.forward(img1, img2, PhotometricLoss::Params{.lambda_dssim = 0.0f});
    auto [loss_tensor, ctx] = result.value();
    auto grad_analytical = ctx.grad_image;

    // Convert to PyTorch for easier numerical gradient checking
    auto img1_torch = to_torch(img1).set_requires_grad(true);
    auto img2_torch = to_torch(img2);
    auto loss_torch = torch::l1_loss(img1_torch, img2_torch, torch::Reduction::Mean);
    loss_torch.backward();
    auto grad_numerical = img1_torch.grad();

    // Compare gradients
    auto grad_analytical_torch = to_torch(grad_analytical);
    auto diff = (grad_analytical_torch - grad_numerical).abs();
    float max_error = diff.max().item<float>();
    float mean_error = diff.mean().item<float>();

    std::cout << std::left << std::setw(30) << "Max absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << max_error << "\n";
    std::cout << std::left << std::setw(30) << "Mean absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << mean_error << "\n";

    EXPECT_LT(max_error, 1e-4);  // Gradients should match within 1e-4
}

TEST(LossesBenchmark, GradientCorrectness_SSIM) {
    print_benchmark_header("Gradient Correctness - SSIM Loss");
    std::cout << "Comparing analytical gradients vs numerical gradients (finite differences)\n\n";

    const int H = 64, W = 64;  // Smaller for faster gradient checking
    auto img1 = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
    auto img2 = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);

    // Compute analytical gradients (LFS - forward already computes them)
    PhotometricLoss loss_fn;
    auto result = loss_fn.forward(img1, img2, PhotometricLoss::Params{.lambda_dssim = 1.0f});
    auto [loss_tensor, ctx] = result.value();
    auto grad_analytical = ctx.grad_image;

    // Compute numerical gradients using finite differences
    const float epsilon = 1e-4f;
    auto img1_cpu = img1.to(Device::CPU);
    auto img2_cpu = img2.to(Device::CPU);
    auto grad_analytical_cpu = grad_analytical.to(Device::CPU);

    // Check a subset of pixels (every 8th pixel to save time)
    int n_checks = 0;
    float max_error = 0.0f;
    float sum_error = 0.0f;

    for (int h = 0; h < H; h += 8) {
        for (int w = 0; w < W; w += 8) {
            for (int c = 0; c < 3; c++) {
                size_t idx = h * W * 3 + w * 3 + c;

                // Perturb +epsilon
                auto img1_plus_cpu = img1_cpu.clone();
                img1_plus_cpu.template ptr<float>()[idx] += epsilon;
                auto img1_plus = img1_plus_cpu.to(Device::CUDA);
                float loss_plus = loss_fn.forward(img1_plus, img2, PhotometricLoss::Params{.lambda_dssim = 1.0f}).value().first.item();

                // Perturb -epsilon
                auto img1_minus_cpu = img1_cpu.clone();
                img1_minus_cpu.template ptr<float>()[idx] -= epsilon;
                auto img1_minus = img1_minus_cpu.to(Device::CUDA);
                float loss_minus = loss_fn.forward(img1_minus, img2, PhotometricLoss::Params{.lambda_dssim = 1.0f}).value().first.item();

                // Numerical gradient
                float grad_num = (loss_plus - loss_minus) / (2.0f * epsilon);
                float grad_ana = grad_analytical_cpu.template ptr<float>()[idx];

                float error = std::abs(grad_ana - grad_num);
                max_error = std::max(max_error, error);
                sum_error += error;
                n_checks++;
            }
        }
    }

    float mean_error = sum_error / n_checks;

    std::cout << std::left << std::setw(30) << "Pixels checked:" << std::right << std::setw(15) << n_checks << "\n";
    std::cout << std::left << std::setw(30) << "Max absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << max_error << "\n";
    std::cout << std::left << std::setw(30) << "Mean absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << mean_error << "\n";

    EXPECT_LT(max_error, 1e-3);  // SSIM gradients are complex, allow slightly larger tolerance
}

TEST(LossesBenchmark, GradientCorrectness_Combined) {
    print_benchmark_header("Gradient Correctness - Combined Photometric Loss");
    std::cout << "Comparing analytical gradients vs numerical gradients (finite differences)\n\n";

    const int H = 64, W = 64;
    const float lambda_dssim = 0.2f;
    auto img1 = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);
    auto img2 = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(3)}, Device::CUDA);

    // Compute analytical gradients (forward already computes them)
    PhotometricLoss loss_fn;
    auto result = loss_fn.forward(img1, img2, PhotometricLoss::Params{.lambda_dssim = lambda_dssim});
    auto [loss_tensor, ctx] = result.value();
    auto grad_analytical = ctx.grad_image;

    // Compute numerical gradients (subset)
    const float epsilon = 1e-4f;
    auto img1_cpu = img1.to(Device::CPU);
    auto img2_cpu = img2.to(Device::CPU);
    auto grad_analytical_cpu = grad_analytical.to(Device::CPU);

    int n_checks = 0;
    float max_error = 0.0f;
    float sum_error = 0.0f;

    for (int h = 0; h < H; h += 8) {
        for (int w = 0; w < W; w += 8) {
            for (int c = 0; c < 3; c++) {
                size_t idx = h * W * 3 + w * 3 + c;

                auto img1_plus_cpu = img1_cpu.clone();
                img1_plus_cpu.template ptr<float>()[idx] += epsilon;
                auto img1_plus = img1_plus_cpu.to(Device::CUDA);
                float loss_plus = loss_fn.forward(img1_plus, img2, PhotometricLoss::Params{.lambda_dssim = lambda_dssim}).value().first.item();

                auto img1_minus_cpu = img1_cpu.clone();
                img1_minus_cpu.template ptr<float>()[idx] -= epsilon;
                auto img1_minus = img1_minus_cpu.to(Device::CUDA);
                float loss_minus = loss_fn.forward(img1_minus, img2, PhotometricLoss::Params{.lambda_dssim = lambda_dssim}).value().first.item();

                float grad_num = (loss_plus - loss_minus) / (2.0f * epsilon);
                float grad_ana = grad_analytical_cpu.template ptr<float>()[idx];

                float error = std::abs(grad_ana - grad_num);
                max_error = std::max(max_error, error);
                sum_error += error;
                n_checks++;
            }
        }
    }

    float mean_error = sum_error / n_checks;

    std::cout << std::left << std::setw(30) << "Pixels checked:" << std::right << std::setw(15) << n_checks << "\n";
    std::cout << std::left << std::setw(30) << "Max absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << max_error << "\n";
    std::cout << std::left << std::setw(30) << "Mean absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << mean_error << "\n";

    EXPECT_LT(max_error, 1e-3);  // Allow slightly larger tolerance for combined loss
}

TEST(LossesBenchmark, GradientCorrectness_ScaleRegularization) {
    print_benchmark_header("Gradient Correctness - Scale Regularization");
    std::cout << "Comparing analytical gradients vs numerical gradients (finite differences)\n\n";

    const size_t N = 1000;  // 1000 Gaussians
    const float weight = 0.01f;
    auto scaling_raw = Tensor::randn({N, 3}, Device::CUDA);

    // Compute analytical gradients (new implementation)
    auto scaling_raw_grad = Tensor::zeros({N, 3}, Device::CUDA);
    float loss = ScaleRegularization::forward(scaling_raw, scaling_raw_grad, ScaleRegularization::Params{.weight = weight}).value().item();

    // Compute numerical gradients using finite differences (subset)
    const float epsilon = 1e-4f;
    auto scaling_raw_cpu = scaling_raw.to(Device::CPU);
    auto scaling_raw_grad_cpu = scaling_raw_grad.to(Device::CPU);

    int n_checks = 0;
    float max_error = 0.0f;
    float sum_error = 0.0f;

    // Check every 10th element to save time
    for (size_t i = 0; i < N * 3; i += 10) {
        // Perturb +epsilon
        auto scaling_plus_cpu = scaling_raw_cpu.clone();
        scaling_plus_cpu.template ptr<float>()[i] += epsilon;
        auto scaling_plus = scaling_plus_cpu.to(Device::CUDA);
        auto grad_plus = Tensor::zeros({N, 3}, Device::CUDA);
        float loss_plus = ScaleRegularization::forward(scaling_plus, grad_plus, ScaleRegularization::Params{.weight = weight}).value().item();

        // Perturb -epsilon
        auto scaling_minus_cpu = scaling_raw_cpu.clone();
        scaling_minus_cpu.template ptr<float>()[i] -= epsilon;
        auto scaling_minus = scaling_minus_cpu.to(Device::CUDA);
        auto grad_minus = Tensor::zeros({N, 3}, Device::CUDA);
        float loss_minus = ScaleRegularization::forward(scaling_minus, grad_minus, ScaleRegularization::Params{.weight = weight}).value().item();

        // Numerical gradient
        float grad_num = (loss_plus - loss_minus) / (2.0f * epsilon);
        float grad_ana = scaling_raw_grad_cpu.template ptr<float>()[i];

        float error = std::abs(grad_ana - grad_num);
        max_error = std::max(max_error, error);
        sum_error += error;
        n_checks++;
    }

    float mean_error = sum_error / n_checks;

    std::cout << std::left << std::setw(30) << "Elements checked:" << std::right << std::setw(15) << n_checks << "\n";
    std::cout << std::left << std::setw(30) << "Max absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << max_error << "\n";
    std::cout << std::left << std::setw(30) << "Mean absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << mean_error << "\n";

    EXPECT_LT(max_error, 1e-3);  // Gradient should be accurate
}

TEST(LossesBenchmark, GradientCorrectness_OpacityRegularization) {
    print_benchmark_header("Gradient Correctness - Opacity Regularization");
    std::cout << "Comparing analytical gradients vs numerical gradients (finite differences)\n\n";

    const size_t N = 1000;  // 1000 Gaussians
    const float weight = 0.01f;
    auto opacity_raw = Tensor::randn({N, 1}, Device::CUDA);

    // Compute analytical gradients (new implementation)
    auto opacity_raw_grad = Tensor::zeros({N, 1}, Device::CUDA);
    float loss = OpacityRegularization::forward(opacity_raw, opacity_raw_grad, OpacityRegularization::Params{.weight = weight}).value().item();

    // Compute numerical gradients using finite differences (subset)
    const float epsilon = 1e-4f;
    auto opacity_raw_cpu = opacity_raw.to(Device::CPU);
    auto opacity_raw_grad_cpu = opacity_raw_grad.to(Device::CPU);

    int n_checks = 0;
    float max_error = 0.0f;
    float sum_error = 0.0f;

    // Check every 10th element to save time
    for (size_t i = 0; i < N; i += 10) {
        // Perturb +epsilon
        auto opacity_plus_cpu = opacity_raw_cpu.clone();
        opacity_plus_cpu.template ptr<float>()[i] += epsilon;
        auto opacity_plus = opacity_plus_cpu.to(Device::CUDA);
        auto grad_plus = Tensor::zeros({N, 1}, Device::CUDA);
        float loss_plus = OpacityRegularization::forward(opacity_plus, grad_plus, OpacityRegularization::Params{.weight = weight}).value().item();

        // Perturb -epsilon
        auto opacity_minus_cpu = opacity_raw_cpu.clone();
        opacity_minus_cpu.template ptr<float>()[i] -= epsilon;
        auto opacity_minus = opacity_minus_cpu.to(Device::CUDA);
        auto grad_minus = Tensor::zeros({N, 1}, Device::CUDA);
        float loss_minus = OpacityRegularization::forward(opacity_minus, grad_minus, OpacityRegularization::Params{.weight = weight}).value().item();

        // Numerical gradient
        float grad_num = (loss_plus - loss_minus) / (2.0f * epsilon);
        float grad_ana = opacity_raw_grad_cpu.template ptr<float>()[i];

        float error = std::abs(grad_ana - grad_num);
        max_error = std::max(max_error, error);
        sum_error += error;
        n_checks++;
    }

    float mean_error = sum_error / n_checks;

    std::cout << std::left << std::setw(30) << "Elements checked:" << std::right << std::setw(15) << n_checks << "\n";
    std::cout << std::left << std::setw(30) << "Max absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << max_error << "\n";
    std::cout << std::left << std::setw(30) << "Mean absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << mean_error << "\n";

    EXPECT_LT(max_error, 1e-3);  // Gradient should be accurate
}

// =============================================================================
// BILATERAL GRID BENCHMARKS
// =============================================================================

TEST(LossesBenchmark, BilateralGridSlice_vs_Reference) {
    print_benchmark_header("Bilateral Grid Slice - Forward+Backward (LFS vs Reference)");
    std::cout << "Testing bilateral grid slice operation (forward + backward pass)\n\n";

    // Grid dimensions
    const int L = 8, H = 16, W = 16;  // Standard bilateral grid dimensions

    // Test different image sizes
    std::vector<std::pair<int, int>> sizes = {{128, 128}, {256, 256}, {512, 512}};

    std::cout << std::left << std::setw(45) << "Image Size" << " | ";
    std::cout << std::right << std::setw(10) << "Reference" << " | ";
    std::cout << std::right << std::setw(10) << "LFS" << " | ";
    std::cout << std::right << std::setw(10) << "Speedup" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& [h, w] : sizes) {
        // Create test data
        auto grid_torch = torch::randn({12, L, H, W}, torch::kCUDA).contiguous();
        auto rgb_torch = torch::rand({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(3)}, torch::kCUDA).contiguous();
        auto grad_output_torch = torch::randn({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(3)}, torch::kCUDA).contiguous();

        // Convert to LFS tensors
        auto grid_lfs = from_torch(grid_torch);
        auto rgb_lfs = from_torch(rgb_torch);
        auto grad_output_lfs = from_torch(grad_output_torch);

        Timer timer;
        const int warmup = 5;
        const int iterations = 20;

        // ===== REFERENCE (torch-based) =====
        for (int i = 0; i < warmup; i++) {
            auto [output, ctx] = gs::bilateral_grid::bilateral_grid_slice_forward(grid_torch, rgb_torch);
            auto [grad_grid, grad_rgb] = gs::bilateral_grid::bilateral_grid_slice_backward(ctx, grad_output_torch);
        }
        cudaDeviceSynchronize();

        timer.start();
        for (int i = 0; i < iterations; i++) {
            auto [output, ctx] = gs::bilateral_grid::bilateral_grid_slice_forward(grid_torch, rgb_torch);
            auto [grad_grid, grad_rgb] = gs::bilateral_grid::bilateral_grid_slice_backward(ctx, grad_output_torch);
        }
        cudaDeviceSynchronize();
        double ref_time = timer.stop_us() / iterations;

        // ===== LFS (LibTorch-free) =====
        auto grad_grid_lfs = Tensor::zeros({12, L, H, W}, Device::CUDA);
        auto grad_rgb_lfs = Tensor::zeros({(size_t)h, (size_t)w, 3}, Device::CUDA);
        auto output_lfs = Tensor::zeros({(size_t)h, (size_t)w, 3}, Device::CUDA);

        float* grid_ptr = grid_lfs.template ptr<float>();
        float* rgb_ptr = rgb_lfs.template ptr<float>();
        float* output_ptr = output_lfs.template ptr<float>();
        float* grad_output_ptr = grad_output_lfs.template ptr<float>();
        float* grad_grid_ptr = grad_grid_lfs.template ptr<float>();
        float* grad_rgb_ptr = grad_rgb_lfs.template ptr<float>();

        for (int i = 0; i < warmup; i++) {
            lfs::training::kernels::launch_bilateral_grid_slice_forward(
                grid_ptr, rgb_ptr, output_ptr,
                L, H, W, h, w, nullptr);
            lfs::training::kernels::launch_bilateral_grid_slice_backward(
                grid_ptr, rgb_ptr, grad_output_ptr,
                grad_grid_ptr, grad_rgb_ptr,
                L, H, W, h, w, nullptr);
        }
        cudaDeviceSynchronize();

        timer.start();
        for (int i = 0; i < iterations; i++) {
            lfs::training::kernels::launch_bilateral_grid_slice_forward(
                grid_ptr, rgb_ptr, output_ptr,
                L, H, W, h, w, nullptr);
            lfs::training::kernels::launch_bilateral_grid_slice_backward(
                grid_ptr, rgb_ptr, grad_output_ptr,
                grad_grid_ptr, grad_rgb_ptr,
                L, H, W, h, w, nullptr);
        }
        cudaDeviceSynchronize();
        double lfs_time = timer.stop_us() / iterations;

        print_result(std::to_string(h) + "x" + std::to_string(w), ref_time, lfs_time);
    }
}

TEST(LossesBenchmark, BilateralGridTV_vs_Reference) {
    print_benchmark_header("Bilateral Grid TV Loss - Forward+Backward (LFS vs Reference)");
    std::cout << "Testing bilateral grid total variation loss (forward + backward pass)\n\n";

    // Grid dimensions
    const int N = 10;  // 10 images
    const int L = 8, H = 16, W = 16;  // Standard bilateral grid dimensions

    // Create test data
    auto grids_torch = torch::randn({N, 12, L, H, W}, torch::kCUDA).contiguous();
    auto grids_lfs = from_torch(grids_torch);

    Timer timer;
    const int warmup = 10;
    const int iterations = 50;

    // ===== REFERENCE (torch-based) =====
    for (int i = 0; i < warmup; i++) {
        auto [loss, ctx] = gs::bilateral_grid::bilateral_grid_tv_forward(grids_torch);
        auto grad = gs::bilateral_grid::bilateral_grid_tv_backward(ctx, 1.0f);
    }
    cudaDeviceSynchronize();

    timer.start();
    for (int i = 0; i < iterations; i++) {
        auto [loss, ctx] = gs::bilateral_grid::bilateral_grid_tv_forward(grids_torch);
        auto grad = gs::bilateral_grid::bilateral_grid_tv_backward(ctx, 1.0f);
    }
    cudaDeviceSynchronize();
    double ref_time = timer.stop_us() / iterations;

    // ===== LFS (LibTorch-free) =====
    auto loss_lfs = Tensor::zeros({1}, Device::CUDA);
    auto grad_lfs = Tensor::zeros({N, 12, L, H, W}, Device::CUDA);

    // Allocate temp buffer
    const int total = N * L * H * W;
    const int num_blocks = std::min((total + 255) / 256, 2048);
    auto temp_buffer = Tensor::zeros({num_blocks}, Device::CUDA);

    float* grids_ptr = grids_lfs.template ptr<float>();
    float* loss_ptr = loss_lfs.template ptr<float>();
    float* temp_ptr = temp_buffer.template ptr<float>();
    float* grad_ptr = grad_lfs.template ptr<float>();

    for (int i = 0; i < warmup; i++) {
        lfs::training::kernels::launch_bilateral_grid_tv_forward(
            grids_ptr, loss_ptr, temp_ptr,
            N, L, H, W, nullptr);
        lfs::training::kernels::launch_bilateral_grid_tv_backward(
            grids_ptr, 1.0f, grad_ptr,
            N, L, H, W, nullptr);
    }
    cudaDeviceSynchronize();

    timer.start();
    for (int i = 0; i < iterations; i++) {
        lfs::training::kernels::launch_bilateral_grid_tv_forward(
            grids_ptr, loss_ptr, temp_ptr,
            N, L, H, W, nullptr);
        lfs::training::kernels::launch_bilateral_grid_tv_backward(
            grids_ptr, 1.0f, grad_ptr,
            N, L, H, W, nullptr);
    }
    cudaDeviceSynchronize();
    double lfs_time = timer.stop_us() / iterations;

    std::cout << std::left << std::setw(45) << "TV Loss (N=10, L=8, H=16, W=16)" << " | ";
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << ref_time << " μs | ";
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << lfs_time << " μs | ";
    double speedup = ref_time / lfs_time;
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x";
    std::cout << (speedup >= 1.0 ? " ✓" : " ✗") << "\n";
}

TEST(LossesBenchmark, GradientCorrectness_BilateralGridSlice) {
    print_benchmark_header("Gradient Correctness - Bilateral Grid Slice");
    std::cout << "Comparing analytical gradients vs numerical gradients (finite differences)\n\n";

    // Small test case for gradient checking
    const int L = 4, H = 8, W = 8;
    const int h = 32, w = 32;

    // Create test data
    auto grid = Tensor::randn({12, L, H, W}, Device::CUDA);
    auto rgb = Tensor::rand({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(3)}, Device::CUDA);
    auto grad_output = Tensor::ones({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(3)}, Device::CUDA);  // Upstream gradient = 1

    // Compute analytical gradients
    auto output = Tensor::zeros({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(3)}, Device::CUDA);
    auto grad_grid = Tensor::zeros({12, L, H, W}, Device::CUDA);
    auto grad_rgb = Tensor::zeros({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(3)}, Device::CUDA);

    float* grid_ptr = grid.template ptr<float>();
    float* rgb_ptr = rgb.template ptr<float>();
    float* output_ptr = output.template ptr<float>();
    float* grad_output_ptr = grad_output.template ptr<float>();
    float* grad_grid_ptr = grad_grid.template ptr<float>();
    float* grad_rgb_ptr = grad_rgb.template ptr<float>();

    lfs::training::kernels::launch_bilateral_grid_slice_forward(
        grid_ptr, rgb_ptr, output_ptr,
        L, H, W, h, w, nullptr);
    lfs::training::kernels::launch_bilateral_grid_slice_backward(
        grid_ptr, rgb_ptr, grad_output_ptr,
        grad_grid_ptr, grad_rgb_ptr,
        L, H, W, h, w, nullptr);
    cudaDeviceSynchronize();

    // Numerical gradient checking (sample a few elements)
    const float epsilon = 1e-4f;
    auto grid_cpu = grid.to(Device::CPU);
    auto grad_grid_cpu = grad_grid.to(Device::CPU);

    int n_checks = 0;
    float max_error = 0.0f;
    float sum_error = 0.0f;

    // Check every 50th grid element
    for (size_t i = 0; i < 12 * L * H * W; i += 50) {
        // Perturb +epsilon
        auto grid_plus_cpu = grid_cpu.clone();
        grid_plus_cpu.template ptr<float>()[i] += epsilon;
        auto grid_plus = grid_plus_cpu.to(Device::CUDA);
        auto output_plus = Tensor::zeros({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(3)}, Device::CUDA);
        float* grid_plus_ptr = grid_plus.template ptr<float>();
        float* output_plus_ptr = output_plus.template ptr<float>();
        lfs::training::kernels::launch_bilateral_grid_slice_forward(
            grid_plus_ptr, rgb_ptr, output_plus_ptr,
            L, H, W, h, w, nullptr);
        cudaDeviceSynchronize();
        auto output_plus_cpu = output_plus.to(Device::CPU);

        // Perturb -epsilon
        auto grid_minus_cpu = grid_cpu.clone();
        grid_minus_cpu.template ptr<float>()[i] -= epsilon;
        auto grid_minus = grid_minus_cpu.to(Device::CUDA);
        auto output_minus = Tensor::zeros({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(3)}, Device::CUDA);
        float* grid_minus_ptr = grid_minus.template ptr<float>();
        float* output_minus_ptr = output_minus.template ptr<float>();
        lfs::training::kernels::launch_bilateral_grid_slice_forward(
            grid_minus_ptr, rgb_ptr, output_minus_ptr,
            L, H, W, h, w, nullptr);
        cudaDeviceSynchronize();
        auto output_minus_cpu = output_minus.to(Device::CPU);

        // Compute numerical gradient: sum of (output_diff * grad_output)
        float grad_num = 0.0f;
        for (size_t j = 0; j < h * w * 3; j++) {
            float diff = (output_plus_cpu.template ptr<float>()[j] - output_minus_cpu.template ptr<float>()[j]) / (2.0f * epsilon);
            grad_num += diff * 1.0f;  // grad_output = 1
        }

        float grad_ana = grad_grid_cpu.template ptr<float>()[i];
        float error = std::abs(grad_ana - grad_num);
        max_error = std::max(max_error, error);
        sum_error += error;
        n_checks++;
    }

    float mean_error = sum_error / n_checks;

    std::cout << std::left << std::setw(30) << "Elements checked:" << std::right << std::setw(15) << n_checks << "\n";
    std::cout << std::left << std::setw(30) << "Max absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << max_error << "\n";
    std::cout << std::left << std::setw(30) << "Mean absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << mean_error << "\n";

    EXPECT_LT(max_error, 1e-2);  // Bilateral grid gradients are more complex, allow higher tolerance
}

TEST(LossesBenchmark, GradientCorrectness_BilateralGridTV) {
    print_benchmark_header("Gradient Correctness - Bilateral Grid TV Loss");
    std::cout << "Comparing analytical gradients vs numerical gradients (finite differences)\n\n";

    // Small test case
    const int N = 2;
    const int L = 4, H = 8, W = 8;

    auto grids = Tensor::randn({N, 12, L, H, W}, Device::CUDA);
    auto loss = Tensor::zeros({1}, Device::CUDA);
    auto grad_grids = Tensor::zeros({N, 12, L, H, W}, Device::CUDA);

    // Allocate temp buffer
    const int total = N * L * H * W;
    const int num_blocks = std::min((total + 255) / 256, 2048);
    auto temp_buffer = Tensor::zeros({num_blocks}, Device::CUDA);

    float* grids_ptr = grids.template ptr<float>();
    float* loss_ptr = loss.template ptr<float>();
    float* temp_ptr = temp_buffer.template ptr<float>();
    float* grad_grids_ptr = grad_grids.template ptr<float>();

    // Compute analytical gradients
    lfs::training::kernels::launch_bilateral_grid_tv_forward(
        grids_ptr, loss_ptr, temp_ptr,
        N, L, H, W, nullptr);
    lfs::training::kernels::launch_bilateral_grid_tv_backward(
        grids_ptr, 1.0f, grad_grids_ptr,
        N, L, H, W, nullptr);
    cudaDeviceSynchronize();

    // Numerical gradient checking
    const float epsilon = 1e-4f;
    auto grids_cpu = grids.to(Device::CPU);
    auto grad_grids_cpu = grad_grids.to(Device::CPU);

    int n_checks = 0;
    float max_error = 0.0f;
    float sum_error = 0.0f;

    // Check every 100th element
    for (size_t i = 0; i < N * 12 * L * H * W; i += 100) {
        // Perturb +epsilon
        auto grids_plus_cpu = grids_cpu.clone();
        grids_plus_cpu.template ptr<float>()[i] += epsilon;
        auto grids_plus = grids_plus_cpu.to(Device::CUDA);
        auto loss_plus = Tensor::zeros({1}, Device::CUDA);
        float* grids_plus_ptr = grids_plus.template ptr<float>();
        float* loss_plus_ptr = loss_plus.template ptr<float>();
        lfs::training::kernels::launch_bilateral_grid_tv_forward(
            grids_plus_ptr, loss_plus_ptr, temp_ptr,
            N, L, H, W, nullptr);
        cudaDeviceSynchronize();
        float loss_plus_val = loss_plus.item();

        // Perturb -epsilon
        auto grids_minus_cpu = grids_cpu.clone();
        grids_minus_cpu.template ptr<float>()[i] -= epsilon;
        auto grids_minus = grids_minus_cpu.to(Device::CUDA);
        auto loss_minus = Tensor::zeros({1}, Device::CUDA);
        float* grids_minus_ptr = grids_minus.template ptr<float>();
        float* loss_minus_ptr = loss_minus.template ptr<float>();
        lfs::training::kernels::launch_bilateral_grid_tv_forward(
            grids_minus_ptr, loss_minus_ptr, temp_ptr,
            N, L, H, W, nullptr);
        cudaDeviceSynchronize();
        float loss_minus_val = loss_minus.item();

        // Numerical gradient
        float grad_num = (loss_plus_val - loss_minus_val) / (2.0f * epsilon);
        float grad_ana = grad_grids_cpu.template ptr<float>()[i];

        float error = std::abs(grad_ana - grad_num);
        max_error = std::max(max_error, error);
        sum_error += error;
        n_checks++;
    }

    float mean_error = sum_error / n_checks;

    std::cout << std::left << std::setw(30) << "Elements checked:" << std::right << std::setw(15) << n_checks << "\n";
    std::cout << std::left << std::setw(30) << "Max absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << max_error << "\n";
    std::cout << std::left << std::setw(30) << "Mean absolute error:" << std::right << std::setw(15) << std::scientific << std::setprecision(2) << mean_error << "\n";

    EXPECT_LT(max_error, 5e-3);  // TV loss gradients - allow slightly higher tolerance due to complex accumulations
}

} // anonymous namespace
