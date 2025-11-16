/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Head-to-head comparison tests: DefaultStrategy (new) vs Reference (old)
// These tests run BOTH implementations on identical inputs and compare outputs

#include "training_new/strategies/default_strategy.hpp"
#include "training/strategies/default_strategy.hpp"
#include "training/rasterization/rasterizer.hpp"
#include "training_new/optimizer/render_output.hpp"
#include "core_new/logger.hpp"
#include "core_new/parameters.hpp"
#include "core_new/splat_data.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

// Helper to convert between tensor types
torch::Tensor to_torch(const lfs::core::Tensor& t) {
    auto cpu_tensor = t.to(lfs::core::Device::CPU);
    std::vector<int64_t> sizes;
    for (size_t i = 0; i < t.ndim(); ++i) {
        sizes.push_back(t.shape()[i]);
    }

    torch::Tensor result;
    if (t.dtype() == lfs::core::DataType::Float32) {
        auto ptr = cpu_tensor.ptr<float>();
        result = torch::from_blob(ptr, sizes, torch::kFloat32).clone();
    } else if (t.dtype() == lfs::core::DataType::Bool) {
        auto ptr = cpu_tensor.ptr<unsigned char>();
        result = torch::from_blob(ptr, sizes, torch::kUInt8).clone().to(torch::kBool);
    }
    return result.cuda();
}

lfs::core::Tensor from_torch(const torch::Tensor& t) {
    auto cpu_t = t.cpu();
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
        auto uint8_t_tensor = cpu_t.to(torch::kUInt8);
        std::vector<uint8_t> data(uint8_t_tensor.data_ptr<uint8_t>(),
                                   uint8_t_tensor.data_ptr<uint8_t>() + uint8_t_tensor.numel());
        // Create on CPU then convert
        auto result = lfs::core::Tensor::zeros_bool(lfs::core::TensorShape(shape), lfs::core::Device::CPU);
        auto ptr = result.ptr<unsigned char>();
        std::copy(data.begin(), data.end(), ptr);
        return result.to(lfs::core::Device::CUDA);
    }

    throw std::runtime_error("Unsupported dtype in from_torch");
}

bool tensors_close(const lfs::core::Tensor& a, const torch::Tensor& b,
                   float rtol = 1e-4f, float atol = 1e-5f) {
    auto a_torch = to_torch(a);
    if (a_torch.sizes() != b.sizes()) {
        std::cout << "Shape mismatch: [";
        for (int i = 0; i < a_torch.dim(); ++i) {
            std::cout << a_torch.size(i) << (i < a_torch.dim()-1 ? ", " : "");
        }
        std::cout << "] vs [";
        for (int i = 0; i < b.dim(); ++i) {
            std::cout << b.size(i) << (i < b.dim()-1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
        return false;
    }

    auto diff = torch::abs(a_torch - b);
    auto threshold = atol + rtol * torch::abs(b);
    auto close = diff <= threshold;

    auto num_close = close.sum().item<int64_t>();
    auto num_total = close.numel();

    if (num_close != num_total) {
        auto max_diff = diff.max().item<float>();
        auto mean_diff = diff.mean().item<float>();
        std::cout << "Tensors not close! " << num_close << "/" << num_total << " elements match" << std::endl;
        std::cout << "Max diff: " << max_diff << ", Mean diff: " << mean_diff << std::endl;

        // Show some example mismatches
        auto not_close_idx = torch::nonzero(~close);
        if (not_close_idx.size(0) > 0) {
            std::cout << "First mismatch at index " << not_close_idx[0].item<int64_t>() << std::endl;
            auto idx = not_close_idx[0].item<int64_t>();
            std::cout << "  New: " << a_torch.flatten()[idx].item<float>() << std::endl;
            std::cout << "  Ref: " << b.flatten()[idx].item<float>() << std::endl;
        }
        return false;
    }

    return true;
}

// Helper to create matching SplatData for both implementations
std::pair<lfs::core::SplatData, gs::SplatData> create_matching_splat_data(int n_gaussians, int seed = 42) {
    torch::manual_seed(seed);

    // Create torch tensors first
    auto torch_means = torch::randn({n_gaussians, 3}, torch::kCUDA) * 0.5f;
    auto torch_sh0 = torch::randn({n_gaussians, 3}, torch::kCUDA) * 0.3f;
    auto torch_shN = torch::randn({n_gaussians, 48}, torch::kCUDA) * 0.1f;
    auto torch_scaling = torch::randn({n_gaussians, 3}, torch::kCUDA) * 0.5f - 2.0f;

    // Identity quaternions
    auto torch_rotation = torch::zeros({n_gaussians, 4}, torch::kCUDA);
    torch_rotation.index({torch::indexing::Slice(), 0}) = 1.0f;

    auto torch_opacity = torch::randn({n_gaussians, 1}, torch::kCUDA) * 0.5f;

    // Convert to new format
    auto lfs_means = from_torch(torch_means);
    auto lfs_sh0 = from_torch(torch_sh0);
    auto lfs_shN = from_torch(torch_shN);
    auto lfs_scaling = from_torch(torch_scaling);
    auto lfs_rotation = from_torch(torch_rotation);
    auto lfs_opacity = from_torch(torch_opacity);

    // Create SplatData objects
    lfs::core::SplatData lfs_splat(3, lfs_means, lfs_sh0, lfs_shN, lfs_scaling, lfs_rotation, lfs_opacity, 1.0f);

    gs::SplatData gs_splat(3, torch_means, torch_sh0, torch_shN, torch_scaling, torch_rotation, torch_opacity, 1.0f);

    return {std::move(lfs_splat), std::move(gs_splat)};
}

// Test 1: Initialization comparison
TEST(DefaultStrategyVsReference, Initialization) {
    auto [lfs_splat, gs_splat] = create_matching_splat_data(100);

    lfs::training::DefaultStrategy lfs_strategy(std::move(lfs_splat));
    gs::training::DefaultStrategy gs_strategy(std::move(gs_splat));

    lfs::core::param::OptimizationParameters lfs_params;
    gs::param::OptimizationParameters gs_params;

    lfs_params.iterations = 1000;
    gs_params.iterations = 1000;

    lfs_strategy.initialize(lfs_params);
    gs_strategy.initialize(gs_params);

    // Compare sizes
    EXPECT_EQ(lfs_strategy.get_model().size(), gs_strategy.get_model().size());

    // Compare parameters (should be identical after initialization)
    EXPECT_TRUE(tensors_close(lfs_strategy.get_model().means(), gs_strategy.get_model().means(), 1e-5f, 1e-6f));
    EXPECT_TRUE(tensors_close(lfs_strategy.get_model().sh0(), gs_strategy.get_model().sh0(), 1e-5f, 1e-6f));
    EXPECT_TRUE(tensors_close(lfs_strategy.get_model().shN(), gs_strategy.get_model().shN(), 1e-5f, 1e-6f));
    EXPECT_TRUE(tensors_close(lfs_strategy.get_model().get_scaling(), gs_strategy.get_model().get_scaling(), 1e-5f, 1e-6f));
    EXPECT_TRUE(tensors_close(lfs_strategy.get_model().get_rotation(), gs_strategy.get_model().get_rotation(), 1e-5f, 1e-6f));
    EXPECT_TRUE(tensors_close(lfs_strategy.get_model().get_opacity(), gs_strategy.get_model().get_opacity(), 1e-5f, 1e-6f));
}

// Test 2: RemoveGaussians comparison
TEST(DefaultStrategyVsReference, RemoveGaussians) {
    auto [lfs_splat, gs_splat] = create_matching_splat_data(50);

    lfs::training::DefaultStrategy lfs_strategy(std::move(lfs_splat));
    gs::training::DefaultStrategy gs_strategy(std::move(gs_splat));

    lfs::core::param::OptimizationParameters lfs_params;
    gs::param::OptimizationParameters gs_params;
    lfs_params.iterations = 100;
    gs_params.iterations = 100;

    lfs_strategy.initialize(lfs_params);
    gs_strategy.initialize(gs_params);

    // Create identical mask
    torch::manual_seed(123);
    auto torch_mask = torch::randint(0, 2, {50}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto lfs_mask = from_torch(torch_mask);

    std::cout << "Removing " << torch_mask.sum().item<int>() << " Gaussians" << std::endl;

    lfs_strategy.remove_gaussians(lfs_mask);
    gs_strategy.remove_gaussians(torch_mask);

    // Compare sizes
    EXPECT_EQ(lfs_strategy.get_model().size(), gs_strategy.get_model().size());

    // Compare remaining parameters
    EXPECT_TRUE(tensors_close(lfs_strategy.get_model().means(), gs_strategy.get_model().means(), 1e-5f, 1e-6f));
    EXPECT_TRUE(tensors_close(lfs_strategy.get_model().sh0(), gs_strategy.get_model().sh0(), 1e-5f, 1e-6f));
    EXPECT_TRUE(tensors_close(lfs_strategy.get_model().get_scaling(), gs_strategy.get_model().get_scaling(), 1e-5f, 1e-6f));
}

// Test 3: Training step with gradient updates
TEST(DefaultStrategyVsReference, TrainingStepWithGradients) {
    auto [lfs_splat, gs_splat] = create_matching_splat_data(30);

    lfs::training::DefaultStrategy lfs_strategy(std::move(lfs_splat));
    gs::training::DefaultStrategy gs_strategy(std::move(gs_splat));

    lfs::core::param::OptimizationParameters lfs_params;
    gs::param::OptimizationParameters gs_params;
    lfs_params.iterations = 100;
    lfs_params.means_lr = 0.001f;
    gs_params.iterations = 100;
    gs_params.means_lr = 0.001f;

    lfs_strategy.initialize(lfs_params);
    gs_strategy.initialize(gs_params);

    // Create identical gradients
    torch::manual_seed(456);
    auto torch_grad = torch::randn({30, 3}, torch::kCUDA) * 0.01f;
    auto lfs_grad = from_torch(torch_grad);

    // Set gradients
    lfs_strategy.get_model().means_grad() = lfs_grad;
    gs_strategy.get_model().means().grad() = torch_grad;

    // Take one step
    lfs_strategy.step(0);
    gs_strategy.step(0);

    // Compare updated parameters
    EXPECT_TRUE(tensors_close(lfs_strategy.get_model().means(), gs_strategy.get_model().means(), 1e-4f, 1e-5f));
}

// Test 4: Full training loop with refinement
TEST(DefaultStrategyVsReference, FullTrainingLoopWithRefinement) {
    const int n_gaussians = 50;
    const int n_iterations = 50;

    auto [lfs_splat, gs_splat] = create_matching_splat_data(n_gaussians);

    lfs::training::DefaultStrategy lfs_strategy(std::move(lfs_splat));
    gs::training::DefaultStrategy gs_strategy(std::move(gs_splat));

    lfs::core::param::OptimizationParameters lfs_params;
    gs::param::OptimizationParameters gs_params;

    lfs_params.iterations = n_iterations;
    lfs_params.start_refine = 10;
    lfs_params.stop_refine = 40;
    lfs_params.refine_every = 10;
    lfs_params.reset_every = 30;
    lfs_params.grad_threshold = 0.0002f;
    lfs_params.sh_degree_interval = 15;

    gs_params.iterations = n_iterations;
    gs_params.start_refine = 10;
    gs_params.stop_refine = 40;
    gs_params.refine_every = 10;
    gs_params.reset_every = 30;
    gs_params.grad_threshold = 0.0002f;
    gs_params.sh_degree_interval = 15;

    lfs_strategy.initialize(lfs_params);
    gs_strategy.initialize(gs_params);

    // Initialize densification info
    lfs_strategy.get_model()._densification_info = lfs::core::Tensor::zeros(
        {2, static_cast<size_t>(n_gaussians)}, lfs::core::Device::CUDA);
    gs_strategy.get_model()._densification_info = torch::zeros({2, n_gaussians}, torch::kCUDA);

    // Create render output structs (we don't use them, just pass them through)
    gs::training::RenderOutput gs_render_output;
    lfs::training::RenderOutput lfs_render_output;

    torch::manual_seed(789);

    for (int iter = 0; iter < n_iterations; ++iter) {
        // Generate identical random gradients
        if (lfs_strategy.get_model().size() == gs_strategy.get_model().size()) {
            int current_size = lfs_strategy.get_model().size();

            auto torch_grad = torch::randn({current_size, 3}, torch::kCUDA) * 0.001f;
            auto lfs_grad = from_torch(torch_grad);

            lfs_strategy.get_model().means_grad() = lfs_grad;
            gs_strategy.get_model().means().grad() = torch_grad;

            // Update densification info with identical values
            auto torch_dens_numer = torch::rand({current_size}, torch::kCUDA) * 0.001f;
            auto torch_dens_denom = torch::ones({current_size}, torch::kCUDA);

            gs_strategy.get_model()._densification_info[0] = torch_dens_denom;
            gs_strategy.get_model()._densification_info[1] = torch_dens_numer;

            lfs_strategy.get_model()._densification_info[0] = from_torch(torch_dens_denom);
            lfs_strategy.get_model()._densification_info[1] = from_torch(torch_dens_numer);
        }

        int lfs_size_before = lfs_strategy.get_model().size();
        int gs_size_before = gs_strategy.get_model().size();

        lfs_strategy.post_backward(iter, lfs_render_output);
        gs_strategy.post_backward(iter, gs_render_output);

        // Resize densification info if size changed
        if (lfs_strategy.get_model().size() != lfs_size_before) {
            lfs_strategy.get_model()._densification_info = lfs::core::Tensor::zeros(
                {2, static_cast<size_t>(lfs_strategy.get_model().size())},
                lfs::core::Device::CUDA);
        }
        if (gs_strategy.get_model().size() != gs_size_before) {
            gs_strategy.get_model()._densification_info = torch::zeros(
                {2, gs_strategy.get_model().size()}, torch::kCUDA);
        }

        lfs_strategy.step(iter);
        gs_strategy.step(iter);

        // Compare sizes after each iteration
        EXPECT_EQ(lfs_strategy.get_model().size(), gs_strategy.get_model().size())
            << "Size mismatch at iteration " << iter;

        // Compare SH degree
        EXPECT_EQ(lfs_strategy.get_model().get_max_sh_degree(), gs_strategy.get_model().get_active_sh_degree())
            << "SH degree mismatch at iteration " << iter;
    }

    // Final comparison
    std::cout << "Final size: " << lfs_strategy.get_model().size() << std::endl;
    std::cout << "Final SH degree - New: " << lfs_strategy.get_model().get_max_sh_degree()
              << ", Ref: " << gs_strategy.get_model().get_active_sh_degree() << std::endl;

    // We can't expect exact numerical match after many operations due to floating point differences
    // But sizes and overall structure should match
    EXPECT_EQ(lfs_strategy.get_model().size(), gs_strategy.get_model().size());
}

// Test 5: Benchmark comparison
TEST(DefaultStrategyVsReference, BenchmarkComparison) {
    const int n_gaussians = 100000;
    const int n_warmup = 3;
    const int n_runs = 10;

    auto [lfs_splat, gs_splat] = create_matching_splat_data(n_gaussians);

    lfs::training::DefaultStrategy lfs_strategy(std::move(lfs_splat));
    gs::training::DefaultStrategy gs_strategy(std::move(gs_splat));

    lfs::core::param::OptimizationParameters lfs_params;
    gs::param::OptimizationParameters gs_params;
    lfs_params.iterations = 100;
    gs_params.iterations = 100;

    // Benchmark initialization
    auto bench_init_lfs = [&]() {
        auto [splat, _] = create_matching_splat_data(n_gaussians);
        lfs::training::DefaultStrategy strategy(std::move(splat));
        lfs::core::param::OptimizationParameters params;
        params.iterations = 100;
        strategy.initialize(params);
    };

    auto bench_init_gs = [&]() {
        auto [_, splat] = create_matching_splat_data(n_gaussians);
        gs::training::DefaultStrategy strategy(std::move(splat));
        gs::param::OptimizationParameters params;
        params.iterations = 100;
        strategy.initialize(params);
    };

    // Warmup
    for (int i = 0; i < n_warmup; ++i) {
        bench_init_lfs();
        bench_init_gs();
    }

    // Time LFS
    cudaDeviceSynchronize();
    auto start_lfs = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_runs; ++i) {
        bench_init_lfs();
    }
    cudaDeviceSynchronize();
    auto end_lfs = std::chrono::high_resolution_clock::now();
    double time_lfs = std::chrono::duration<double, std::milli>(end_lfs - start_lfs).count() / n_runs;

    // Time reference
    cudaDeviceSynchronize();
    auto start_gs = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_runs; ++i) {
        bench_init_gs();
    }
    cudaDeviceSynchronize();
    auto end_gs = std::chrono::high_resolution_clock::now();
    double time_gs = std::chrono::duration<double, std::milli>(end_gs - start_gs).count() / n_runs;

    std::cout << "\n=== BENCHMARK COMPARISON ===" << std::endl;
    std::cout << "Initialization (" << n_gaussians << " Gaussians):" << std::endl;
    std::cout << "  New (LibTorch-free): " << time_lfs << " ms" << std::endl;
    std::cout << "  Reference (LibTorch): " << time_gs << " ms" << std::endl;
    std::cout << "  Speedup: " << (time_gs / time_lfs) << "x" << std::endl;

    // We expect the new implementation to be at least as fast
    // Give some tolerance for measurement noise
    EXPECT_LT(time_lfs, time_gs * 1.5) << "New implementation should not be significantly slower";
}
