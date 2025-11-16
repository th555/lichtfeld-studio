/* Test the EXACT pattern from densification */
#include "core_new/tensor.hpp"
#include "core_new/splat_data.hpp"
#include "core/splat_data.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>

using namespace lfs::core;

// Helper to convert torch to lfs tensor  
Tensor from_torch(const torch::Tensor& t) {
    auto cpu_t = t.cpu().contiguous();
    std::vector<size_t> shape;
    for (int64_t i = 0; i < cpu_t.dim(); ++i) {
        shape.push_back(cpu_t.size(i));
    }
    if (cpu_t.dtype() == torch::kFloat32) {
        std::vector<float> data(cpu_t.data_ptr<float>(),
                                 cpu_t.data_ptr<float>() + cpu_t.numel());
        return Tensor::from_vector(data, TensorShape(shape), Device::CUDA);
    }
    throw std::runtime_error("Unsupported dtype");
}

TEST(ExactDensificationPattern, TorchFirst_Then_LFS_SplatData) {
    std::cout << "\n=== EXACT DENSIFICATION PATTERN ===" << std::endl;
    
    // Step 1: Create and use LibTorch SplatData (like reference impl)
    std::cout << "Step 1: Creating LibTorch SplatData (100K)..." << std::endl;
    torch::manual_seed(999);
    auto torch_rotation_ref = torch::zeros({100000, 4}, torch::kCUDA);
    torch_rotation_ref.index({torch::indexing::Slice(), 0}) = 1.0f;
    
    gs::SplatData gs_splat(3, 
        torch::randn({100000, 3}, torch::kCUDA),
        torch::randn({100000, 3}, torch::kCUDA),
        torch::randn({100000, 48}, torch::kCUDA),
        torch::randn({100000, 3}, torch::kCUDA) - 2.0f,
        torch_rotation_ref,
        torch::randn({100000, 1}, torch::kCUDA),
        1.0f);
    
    // Use it (like reference densification)
    std::cout << "Step 2: Using LibTorch SplatData..." << std::endl;
    auto ref_rotation = gs_splat.get_rotation();
    std::cout << "  LibTorch get_rotation() succeeded" << std::endl;
    
    // Step 3: Now create LFS SplatData (like new impl)
    std::cout << "Step 3: Creating LFS SplatData (100K)..." << std::endl;
    torch::manual_seed(999);
    auto torch_rotation = torch::zeros({100000, 4}, torch::kCUDA);
    torch_rotation.index({torch::indexing::Slice(), 0}) = 1.0f;
    
    auto lfs_rotation = from_torch(torch_rotation);
    lfs::core::SplatData lfs_splat(3,
        from_torch(torch::randn({100000, 3}, torch::kCUDA)),
        from_torch(torch::randn({100000, 3}, torch::kCUDA)),
        from_torch(torch::randn({100000, 48}, torch::kCUDA)),
        from_torch(torch::randn({100000, 3}, torch::kCUDA) - 2.0f),
        lfs_rotation,
        from_torch(torch::randn({100000, 1}, torch::kCUDA)),
        1.0f);
    
    // Step 4: NOW try to use LFS SplatData (THIS IS WHERE IT FAILS)
    std::cout << "Step 4: Using LFS SplatData get_rotation()..." << std::endl;
    try {
        auto lfs_rot = lfs_splat.get_rotation();
        std::cout << "  LFS get_rotation() SUCCESS!" << std::endl;
        SUCCEED();
    } catch (const std::exception& e) {
        std::cout << "  LFS get_rotation() FAILED: " << e.what() << std::endl;
        FAIL() << "LFS get_rotation() failed after LibTorch usage: " << e.what();
    }
}
/* Test LFS densification WITHOUT LibTorch reference */
#include "core_new/tensor.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/parameters.hpp"
#include "training_new/strategies/default_strategy.hpp"
#include "optimizer/render_output.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace lfs::core;

TEST(LFSOnlyDensification, WorksAt100K) {
    std::cout << "\n=== LFS-ONLY DENSIFICATION TEST (NO LIBTORCH) ===" << std::endl;

    // CRITICAL: Initialize CUDA runtime FIRST to populate device count caches
    // This must happen before any other CUDA operations to ensure CUB's
    // static device count cache is correctly initialized
    int device_count = -1;
    cudaGetDeviceCount(&device_count);
    printf("[TEST] CUDA device count: %d\n", device_count);

    cudaSetDevice(0);
    int current_device = -1;
    cudaGetDevice(&current_device);
    printf("[TEST] Current CUDA device: %d\n", current_device);

    // Force CUDA runtime initialization with a dummy allocation
    void* dummy_ptr = nullptr;
    cudaMalloc(&dummy_ptr, 1024);
    cudaFree(dummy_ptr);
    printf("[TEST] CUDA runtime initialized\n");

    // Test with 10M Gaussians - should use ~2.48GB, ~4.96GB after split
    // This is a realistic scenario that should NOT OOM on a 24GB GPU
    std::vector<int> gaussian_counts = {10000000};

    for (int n_gaussians : gaussian_counts) {
        std::cout << "\nTesting with " << n_gaussians << " Gaussians..." << std::endl;

        // Create LFS SplatData (NO LIBTORCH!)
        auto rotation = Tensor::zeros({n_gaussians, 4}, Device::CUDA);
        rotation.slice(1, 0, 1).fill_(1.0f);  // w=1, x=y=z=0

        lfs::core::SplatData splat(3,
            Tensor::randn({n_gaussians, 3}, Device::CUDA),
            Tensor::randn({n_gaussians, 3}, Device::CUDA),
            Tensor::randn({n_gaussians, 48}, Device::CUDA),
            Tensor::randn({n_gaussians, 3}, Device::CUDA) - 2.0f,
            rotation,
            Tensor::randn({n_gaussians, 1}, Device::CUDA),
            1.0f);

        // Initialize densification_info with high gradients
        // Shape: [2, n_gaussians] - row 0 is denom (count), row 1 is numer (accumulated grads)
        splat._densification_info = Tensor::ones({2, static_cast<size_t>(n_gaussians)}, Device::CUDA);
        // Set row 1 to high values (accumulated gradients)
        auto numer = Tensor::ones({static_cast<size_t>(n_gaussians)}, Device::CUDA) * 10.0f;
        splat._densification_info[1] = numer;

        int size_before = splat.size();
        std::cout << "Before: " << size_before << std::endl;

        // Create strategy
        lfs::training::DefaultStrategy strat(std::move(splat));
        lfs::core::param::OptimizationParameters params;
        params.iterations = 30000;
        params.start_refine = 500;
        params.refine_every = 100;
        params.stop_refine = 15000;
        params.grad_threshold = 0.0002f;  // Default threshold
        params.grow_scale3d = 0.01f;      // Default grow threshold
        params.prune_scale3d = 0.15f;     // Default prune threshold
        params.prune_opacity = 0.005f;    // Default opacity threshold
        params.reset_every = 3000;
        params.pause_refine_after_reset = 0;
        params.sh_degree_interval = 1000;
        strat.initialize(params);

        // Get model reference and re-initialize densification info (initialize() resets it)
        strat.get_model()._densification_info = Tensor::ones({2, static_cast<size_t>(n_gaussians)}, Device::CUDA);
        // Set row 1 to high values (accumulated gradients)
        auto numer2 = Tensor::ones({static_cast<size_t>(n_gaussians)}, Device::CUDA) * 10.0f;
        strat.get_model()._densification_info[1] = numer2;

        // Create fake render output (not used by current implementation)
        lfs::training::RenderOutput render_output;

        int test_iter = 600;  // Within refine range (500 < 600 < 15000, divisible by 100)

        // Run densification
        try {
            strat.post_backward(test_iter, render_output);
            int size_after = strat.get_model().size();
            int added = size_after - size_before;
            std::cout << "After: " << size_after << " (+=" << added << ")" << std::endl;

            // Should add Gaussians
            EXPECT_GT(added, 0) << "Should add Gaussians at " << n_gaussians;
            // Don't expect exact doubling, just that Gaussians were added

            std::cout << "✓ SUCCESS at " << n_gaussians << " Gaussians" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "✗ FAILED at " << n_gaussians << " Gaussians: " << e.what() << std::endl;
            FAIL() << "LFS-only densification failed: " << e.what();
        }
    }
}
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Densification test: Compare reference and new implementations

#include "training_new/strategies/default_strategy.hpp"
#include "training/strategies/default_strategy.hpp"
#include "core_new/splat_data.hpp"
#include "core/splat_data.hpp"
#include "training_new/optimizer/render_output.hpp"
#include "training/rasterization/rasterizer.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <chrono>

namespace {

// Helper to convert torch to lfs tensor
lfs::core::Tensor from_torch(const torch::Tensor& t) {
    auto cpu_t = t.cpu().contiguous();
    std::vector<size_t> shape;
    for (int64_t i = 0; i < cpu_t.dim(); ++i) {
        shape.push_back(cpu_t.size(i));
    }

    if (cpu_t.dtype() == torch::kFloat32) {
        std::vector<float> data(cpu_t.data_ptr<float>(),
                                 cpu_t.data_ptr<float>() + cpu_t.numel());
        return lfs::core::Tensor::from_vector(data, lfs::core::TensorShape(shape),
                                               lfs::core::Device::CUDA);
    } else if (cpu_t.dtype() == torch::kBool) {
        auto uint8_tensor = cpu_t.to(torch::kUInt8);
        auto result = lfs::core::Tensor::zeros_bool(lfs::core::TensorShape(shape), lfs::core::Device::CPU);
        auto ptr = result.ptr<unsigned char>();
        std::memcpy(ptr, uint8_tensor.data_ptr<uint8_t>(), uint8_tensor.numel());
        return result.to(lfs::core::Device::CUDA);
    }
    throw std::runtime_error("Unsupported dtype");
}

// Create matching splat data with realistic parameters
std::pair<lfs::core::SplatData, gs::SplatData> create_matching_data(int n, int seed = 42) {
    torch::manual_seed(seed);

    auto torch_means = torch::randn({n, 3}, torch::kCUDA) * 0.5f;
    auto torch_sh0 = torch::randn({n, 3}, torch::kCUDA) * 0.3f;
    auto torch_shN = torch::randn({n, 48}, torch::kCUDA) * 0.1f;
    auto torch_scaling = torch::randn({n, 3}, torch::kCUDA) * 0.5f - 2.0f;
    auto torch_rotation = torch::zeros({n, 4}, torch::kCUDA);
    torch_rotation.index({torch::indexing::Slice(), 0}) = 1.0f;
    auto torch_opacity = torch::randn({n, 1}, torch::kCUDA) * 0.5f;

    auto lfs_means = from_torch(torch_means);
    auto lfs_sh0 = from_torch(torch_sh0);
    auto lfs_shN = from_torch(torch_shN);
    auto lfs_scaling = from_torch(torch_scaling);
    auto lfs_rotation = from_torch(torch_rotation);
    auto lfs_opacity = from_torch(torch_opacity);

    lfs::core::SplatData lfs_splat(3, lfs_means, lfs_sh0, lfs_shN, lfs_scaling, lfs_rotation, lfs_opacity, 1.0f);
    gs::SplatData gs_splat(3, torch_means, torch_sh0, torch_shN, torch_scaling, torch_rotation, torch_opacity, 1.0f);

    return {std::move(lfs_splat), std::move(gs_splat)};
}

// Simulate realistic densification info (accumulated gradients + counts)
void setup_densification_info(lfs::core::SplatData& lfs_splat, gs::SplatData& gs_splat,
                                float grad_mean = 0.0005f, float grad_std = 0.0003f) {
    int n = lfs_splat.size();

    // Create matching densification info for both implementations
    torch::manual_seed(123);
    auto torch_dinfo = torch::randn({2, n}, torch::kCUDA) * grad_std + grad_mean;
    torch_dinfo = torch_dinfo.abs();  // Gradients are positive

    // densification_info layout: [0] = denom (counts), [1] = numer (accumulated grads)
    torch_dinfo.index({0, torch::indexing::Slice()}) = 100.0f;  // Set counts to 100
    // Index 1 already has the accumulated gradients from randn above

    auto lfs_dinfo = from_torch(torch_dinfo);

    lfs_splat._densification_info = lfs_dinfo;
    gs_splat._densification_info = torch_dinfo;
}

}  // anonymous namespace

// Test densification comparison
TEST(DensificationBenchmark, CompareImplementations) {
    std::vector<int> gaussian_counts = {10000, 100000, 500000, 1000000, 2000000};  // Scale up

    std::cout << "\n=== DENSIFICATION COMPARISON (REF vs NEW) ===" << std::endl;

    for (int n_gaussians : gaussian_counts) {
        std::cout << "Testing with " << n_gaussians << " Gaussians..." << std::endl;

        // Create matching data
        auto [lfs_splat, gs_splat] = create_matching_data(n_gaussians, 999);

        lfs::training::DefaultStrategy lfs_strat(std::move(lfs_splat));
        gs::training::DefaultStrategy gs_strat(std::move(gs_splat));

        lfs::core::param::OptimizationParameters lfs_params;
        gs::param::OptimizationParameters gs_params;

        lfs_params.iterations = 30000;
        lfs_params.start_refine = 500;
        lfs_params.refine_every = 100;
        lfs_params.stop_refine = 15000;
        lfs_params.grad_threshold = 0.0002f;

        gs_params.iterations = 30000;
        gs_params.start_refine = 500;
        gs_params.refine_every = 100;
        gs_params.stop_refine = 15000;
        gs_params.grad_threshold = 0.0002f;

        lfs_strat.initialize(lfs_params);
        gs_strat.initialize(gs_params);

        // Setup realistic densification info (simulate accumulated gradients)
        setup_densification_info(lfs_strat.get_model(), gs_strat.get_model(), 5.0f, 3.0f);

        // Create dummy render outputs
        lfs::training::RenderOutput lfs_render_output;
        gs::training::RenderOutput gs_render_output;

        // Use iteration 600 (within refinement window, will trigger densification)
        int test_iter = 600;

        int size_before_lfs = lfs_strat.get_model().size();
        int size_before_gs = gs_strat.get_model().size();

        std::cout << "Before - LFS: " << size_before_lfs << ", REF: " << size_before_gs << std::endl;

        // Run reference first
        std::cout << "\n=== RUNNING REFERENCE ===" << std::endl;
        gs_strat.post_backward(test_iter, gs_render_output);
        int size_after_gs = gs_strat.get_model().size();
        std::cout << "After REF: " << size_after_gs << " (+=" << (size_after_gs - size_before_gs) << ")" << std::endl;

        // Debug: Check CUDA state after LibTorch
        std::cout << "\n=== CHECKING CUDA STATE AFTER LIBTORCH ===" << std::endl;
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        std::cout << "Device count: " << device_count << std::endl;

        int current_device = -1;
        cudaGetDevice(&current_device);
        std::cout << "Current device: " << current_device << std::endl;

        // Check if there are any CUDA errors pending
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "Pending CUDA error: " << cudaGetErrorString(err) << std::endl;
            cudaGetLastError(); // Clear it
        }

        // Run new implementation
        std::cout << "\n=== RUNNING NEW ===" << std::endl;
        try {
            lfs_strat.post_backward(test_iter, lfs_render_output);
            int size_after_lfs = lfs_strat.get_model().size();
            std::cout << "After NEW: " << size_after_lfs << " (+=" << (size_after_lfs - size_before_lfs) << ")" << std::endl;

            // Verify both implementations grew the same amount
            EXPECT_EQ(size_after_lfs - size_before_lfs, size_after_gs - size_before_gs)
                << "Different growth behavior for " << n_gaussians << " Gaussians";
        } catch (const std::exception& e) {
            std::cout << "NEW IMPLEMENTATION ERROR: " << e.what() << std::endl;
            FAIL() << "New implementation threw exception: " << e.what();
        }
    }
}
/* Profile duplicate and split operations separately */
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <iostream>
#include "core_new/tensor.hpp"
#include "core_new/splat_data.hpp"
#include "training_new/strategies/default_strategy.hpp"
#include "optimizer/render_output.hpp"
#include "core_new/parameters.hpp"

using namespace lfs::core;

class Timer {
public:
    void start() {
        cudaDeviceSynchronize();
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

TEST(ProfileOps, DuplicateOnly) {
    std::cout << "\n=== PROFILE DUPLICATE OPERATION ===" << std::endl;


    const int n = 10000000;
    std::cout << "Gaussians to duplicate: " << static_cast<double>(n) << std::endl;

    Timer timer;
    double t;

    // Create initial model
    auto rotation = Tensor::zeros({n, 4}, Device::CUDA);
    rotation.slice(1, 0, 1).fill_(1.0f);

    lfs::core::SplatData splat(3,
        Tensor::randn({n, 3}, Device::CUDA),
        Tensor::randn({n, 3}, Device::CUDA),
        Tensor::randn({n, 48}, Device::CUDA),
        Tensor::randn({n, 3}, Device::CUDA) - 2.0f,
        rotation,
        Tensor::randn({n, 1}, Device::CUDA),
        1.0f);

    splat._densification_info = Tensor::ones({2, static_cast<size_t>(n)}, Device::CUDA);
    auto numer = Tensor::ones({static_cast<size_t>(n)}, Device::CUDA) * 10.0f;
    splat._densification_info[1] = numer;

    lfs::training::DefaultStrategy strat(std::move(splat));
    lfs::core::param::OptimizationParameters params;
    params.grad_threshold = 0.0002f;
    params.grow_scale3d = 100.0f;  // Make everything "small" so all duplicate
    strat.initialize(params);

    strat.get_model()._densification_info = Tensor::ones({2, static_cast<size_t>(n)}, Device::CUDA);
    auto numer2 = Tensor::ones({static_cast<size_t>(n)}, Device::CUDA) * 10.0f;
    strat.get_model()._densification_info[1] = numer2;

    // Compute masks
    Tensor numer_grad = strat.get_model()._densification_info[1];
    Tensor denom_grad = strat.get_model()._densification_info[0];
    const Tensor grads = numer_grad / denom_grad.clamp_min(1.0f);
    const Tensor is_grad_high = grads > params.grad_threshold;
    const Tensor max_values = strat.get_model().get_scaling().max(-1, false);
    const Tensor is_small = max_values <= (params.grow_scale3d * strat.get_model().get_scene_scale());
    const Tensor is_duplicated = is_grad_high.logical_and(is_small);

    std::cout << "Gaussians to duplicate: " << is_duplicated.sum_scalar() << std::endl;

    // Time grow_gs which will only do duplicate since everything is "small"
    timer.start();
    lfs::training::RenderOutput render_output;
    strat.post_backward(600, render_output);
    t = timer.stop();

    std::cout << "post_backward (duplicate only) time: " << t << " ms" << std::endl;
    std::cout << "Result: " << n << " -> " << strat.get_model().size() << std::endl;
}

TEST(ProfileOps, SplitOnly) {
    std::cout << "\n=== PROFILE SPLIT OPERATION ===" << std::endl;


    const int n = 10000000;
    std::cout << "Gaussians to split: " << static_cast<double>(n) << std::endl;

    Timer timer;
    double t;

    // Create initial model
    auto rotation = Tensor::zeros({n, 4}, Device::CUDA);
    rotation.slice(1, 0, 1).fill_(1.0f);

    lfs::core::SplatData splat(3,
        Tensor::randn({n, 3}, Device::CUDA),
        Tensor::randn({n, 3}, Device::CUDA),
        Tensor::randn({n, 48}, Device::CUDA),
        Tensor::randn({n, 3}, Device::CUDA) - 2.0f,
        rotation,
        Tensor::randn({n, 1}, Device::CUDA),
        1.0f);

    splat._densification_info = Tensor::ones({2, static_cast<size_t>(n)}, Device::CUDA);
    auto numer = Tensor::ones({static_cast<size_t>(n)}, Device::CUDA) * 10.0f;
    splat._densification_info[1] = numer;

    lfs::training::DefaultStrategy strat(std::move(splat));
    lfs::core::param::OptimizationParameters params;
    params.grad_threshold = 0.0002f;
    params.grow_scale3d = 0.0f;  // Make everything "large" so all split
    strat.initialize(params);

    strat.get_model()._densification_info = Tensor::ones({2, static_cast<size_t>(n)}, Device::CUDA);
    auto numer2 = Tensor::ones({static_cast<size_t>(n)}, Device::CUDA) * 10.0f;
    strat.get_model()._densification_info[1] = numer2;

    // Compute masks
    Tensor numer_grad = strat.get_model()._densification_info[1];
    Tensor denom_grad = strat.get_model()._densification_info[0];
    const Tensor grads = numer_grad / denom_grad.clamp_min(1.0f);
    const Tensor is_grad_high = grads > params.grad_threshold;
    const Tensor max_values = strat.get_model().get_scaling().max(-1, false);
    const Tensor is_small = max_values <= (params.grow_scale3d * strat.get_model().get_scene_scale());
    const Tensor is_large = is_small.logical_not();
    Tensor is_split = is_grad_high.logical_and(is_large);

    std::cout << "Gaussians to split: " << is_split.sum_scalar() << std::endl;

    // Time grow_gs which will only do split since everything is "large"
    timer.start();
    lfs::training::RenderOutput render_output;
    strat.post_backward(600, render_output);
    t = timer.stop();

    std::cout << "post_backward (split only) time: " << t << " ms" << std::endl;
    std::cout << "Result: " << n << " -> " << strat.get_model().size() << std::endl;
}
// Test OLD LibTorch-based densification at 10M scale
// Uses src/training/strategies/default_strategy.cpp

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "core/splat_data.hpp"
#include "training/strategies/default_strategy.hpp"
#include "training/rasterization/rasterizer.hpp"
#include "core/parameters.hpp"

TEST(OldDensification, WorksAt10M) {
    std::cout << "\n=== OLD LIBTORCH-BASED DENSIFICATION TEST ===\n";

    const int N = 10000000;  // 10M Gaussians

    std::cout << "Testing with " << N << " Gaussians...\n";

    // Create SplatData with test data
    auto positions = torch::randn({N, 3}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto rotations = torch::randn({N, 4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    // Make scales SMALL so they qualify for duplication (grow_scale3d * scene_scale = 0.01 * 1.0 = 0.01)
    auto scales = torch::rand({N, 3}, torch::device(torch::kCUDA).dtype(torch::kFloat32)) * 0.005f;  // All < 0.01
    auto sh0 = torch::randn({N, 3}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto shN = torch::randn({N, 45}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto opacities = torch::randn({N, 1}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // Create SplatData (OLD API requires sh_degree as first arg)
    gs::SplatData splat_data(
        3,  // sh_degree
        positions,
        sh0,
        shN,
        scales,
        rotations,
        opacities,
        1.0f  // scene_scale
    );

    // Create strategy
    gs::training::DefaultStrategy strategy(std::move(splat_data));

    // Create minimal parameters
    gs::param::OptimizationParameters params;
    params.grad_threshold = 0.0002f;
    params.grow_scale3d = 0.01f;
    params.prune_opacity = 0.005f;
    params.prune_scale3d = 0.1f;
    params.start_refine = 500;
    params.refine_every = 100;
    params.pause_refine_after_reset = 0;
    params.reset_every = 3000;
    params.stop_refine = 15000;
    params.revised_opacity = false;

    strategy.initialize(params);

    // Re-initialize densification info AFTER initialize() (which resets it)
    strategy.get_model()._densification_info = torch::ones({2, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    strategy.get_model()._densification_info[1] = torch::ones({N}, torch::device(torch::kCUDA)) * 10.0f;  // High gradients

    int before = strategy.get_model().means().size(0);
    std::cout << "Before: " << before << "\n";

    // Create a fake RenderOutput (needed for post_backward)
    gs::training::RenderOutput render_output;
    render_output.image = torch::zeros({3, 800, 800}, torch::device(torch::kCUDA));
    render_output.depth = torch::zeros({1, 800, 800}, torch::device(torch::kCUDA));
    render_output.alpha = torch::zeros({1, 800, 800}, torch::device(torch::kCUDA));
    render_output.radii = torch::zeros({N}, torch::device(torch::kCUDA).dtype(torch::kInt32));

    // Time the densification operation
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    // Run densification through post_backward (which calls grow_gs internally)
    strategy.post_backward(1000, render_output);  // Iteration 1000

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    int after = strategy.get_model().means().size(0);
    std::cout << "After: " << after << " (+=" << (after - before) << ")\n";
    std::cout << "Time: " << duration << " ms\n";

    EXPECT_GT(after, before);
    std::cout << "✓ SUCCESS at " << N << " Gaussians\n";
}
/* Test OLD LibTorch-based densification - EXACT same test as LFSOnlyDensification */
#include "core/splat_data.hpp"
#include "core/parameters.hpp"
#include "training/strategies/default_strategy.hpp"
#include "training/rasterization/rasterizer.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <chrono>

TEST(OldLibTorchDensification, WorksAt10M) {
    std::cout << "\n=== OLD LIBTORCH DENSIFICATION TEST (10M) ===" << std::endl;

    // CRITICAL: Initialize CUDA runtime FIRST
    int device_count = -1;
    cudaGetDeviceCount(&device_count);
    printf("[TEST] CUDA device count: %d\n", device_count);

    cudaSetDevice(0);
    int current_device = -1;
    cudaGetDevice(&current_device);
    printf("[TEST] Current CUDA device: %d\n", current_device);

    // Force CUDA runtime initialization
    void* dummy_ptr = nullptr;
    cudaMalloc(&dummy_ptr, 1024);
    cudaFree(dummy_ptr);
    printf("[TEST] CUDA runtime initialized\n");

    const int n_gaussians = 10000000;  // 10M Gaussians

    std::cout << "\nTesting with " << n_gaussians << " Gaussians..." << std::endl;

    // Create OLD LibTorch-based SplatData
    auto rotation = torch::zeros({n_gaussians, 4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    rotation.slice(1, 0, 1).fill_(1.0f);  // w=1, x=y=z=0

    auto positions = torch::randn({n_gaussians, 3}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto sh0 = torch::randn({n_gaussians, 3}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto shN = torch::randn({n_gaussians, 48}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto scales = torch::randn({n_gaussians, 3}, torch::device(torch::kCUDA).dtype(torch::kFloat32)) - 2.0f;
    auto opacities = torch::randn({n_gaussians, 1}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    gs::SplatData splat(
        3,  // sh_degree
        positions,
        sh0,
        shN,
        scales,
        rotation,
        opacities,
        1.0f  // scene_scale
    );

    // Initialize densification_info with high gradients
    splat._densification_info = torch::ones({2, n_gaussians}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    // Set row 1 to high values (accumulated gradients)
    splat._densification_info[1] = torch::ones({n_gaussians}, torch::device(torch::kCUDA)) * 10.0f;

    int size_before = splat.means().size(0);
    std::cout << "Before: " << size_before << std::endl;

    // Create strategy
    gs::training::DefaultStrategy strat(std::move(splat));
    gs::param::OptimizationParameters params;
    params.iterations = 30000;
    params.start_refine = 500;
    params.refine_every = 100;
    params.stop_refine = 15000;
    params.grad_threshold = 0.0002f;
    params.grow_scale3d = 0.01f;
    params.prune_scale3d = 0.15f;
    params.prune_opacity = 0.005f;
    params.reset_every = 3000;
    params.pause_refine_after_reset = 0;
    params.sh_degree_interval = 1000;
    strat.initialize(params);

    // Re-initialize densification info (initialize() resets it)
    strat.get_model()._densification_info = torch::ones({2, n_gaussians}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    strat.get_model()._densification_info[1] = torch::ones({n_gaussians}, torch::device(torch::kCUDA)) * 10.0f;

    // Create fake render output
    gs::training::RenderOutput render_output;
    render_output.image = torch::zeros({3, 800, 800}, torch::device(torch::kCUDA));
    render_output.depth = torch::zeros({1, 800, 800}, torch::device(torch::kCUDA));
    render_output.alpha = torch::zeros({1, 800, 800}, torch::device(torch::kCUDA));
    render_output.radii = torch::zeros({n_gaussians}, torch::device(torch::kCUDA).dtype(torch::kInt32));

    int test_iter = 600;  // Within refine range (500 < 600 < 15000, divisible by 100)

    // Run densification and TIME it
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    try {
        strat.post_backward(test_iter, render_output);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        int size_after = strat.get_model().means().size(0);
        int added = size_after - size_before;
        std::cout << "After: " << size_after << " (+=" << added << ")" << std::endl;
        std::cout << "Time: " << duration << " ms" << std::endl;

        // Should add Gaussians
        EXPECT_GT(added, 0) << "Should add Gaussians at " << n_gaussians;

        std::cout << "✓ SUCCESS at " << n_gaussians << " Gaussians" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED at " << n_gaussians << " Gaussians: " << e.what() << std::endl;
        FAIL() << "OLD LibTorch densification failed: " << e.what();
    }
}
