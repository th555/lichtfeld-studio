/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_add_new_gs_deterministic.cpp
 * @brief Deterministic comparison of add_new_gs between legacy and new implementations
 *
 * This test compares the add_new_gs operation WITHOUT any stochastic influences:
 * - No multinomial sampling (we manually specify indices)
 * - No training gradients
 * - No noise injection
 * - Multiple iterations to verify optimizer state handling
 *
 * This isolates the core add_new_gs logic from all sources of randomness.
 */

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

// Legacy implementation
#include "training/strategies/mcmc.hpp"
#include "core/splat_data.hpp"
#include "core/parameters.hpp"

// New implementation
#include "training_new/strategies/mcmc.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/parameters.hpp"
#include "core_new/tensor.hpp"

using namespace lfs::core;

class AddNewGsDeterministicTest : public ::testing::Test {
protected:
    void SetUp() override {
        spdlog::set_level(spdlog::level::info);

        // Set seeds for reproducibility
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }

    // Helper: Compare tensors between torch and lfs
    void compareTensors(const torch::Tensor& legacy_tensor,
                       const Tensor& new_tensor,
                       const std::string& name,
                       float tolerance = 1e-5f) {
        auto legacy_cpu = legacy_tensor.cpu().contiguous();
        auto new_cpu = new_tensor.cpu().contiguous();

        ASSERT_EQ(legacy_cpu.numel(), new_cpu.numel())
            << name << " numel mismatch";

        const float* legacy_ptr = legacy_cpu.data_ptr<float>();
        const float* new_ptr = new_cpu.ptr<float>();
        size_t numel = legacy_cpu.numel();

        float max_diff = 0.0f;
        float mean_diff = 0.0f;
        size_t num_mismatches = 0;

        for (size_t i = 0; i < numel; ++i) {
            float diff = std::abs(legacy_ptr[i] - new_ptr[i]);
            max_diff = std::max(max_diff, diff);
            mean_diff += diff;

            if (diff > tolerance) {
                num_mismatches++;
                if (num_mismatches <= 5) {
                    spdlog::warn("{} mismatch at index {}: legacy={:.8f}, new={:.8f}, diff={:.8e}",
                                name, i, legacy_ptr[i], new_ptr[i], diff);
                }
            }
        }

        mean_diff /= numel;

        spdlog::info("{}: max_diff={:.8e}, mean_diff={:.8e}, mismatches={}/{} ({:.2f}%)",
                    name, max_diff, mean_diff, num_mismatches, numel,
                    100.0f * num_mismatches / numel);

        EXPECT_LT(max_diff, tolerance) << name << " max difference exceeds tolerance";
        EXPECT_EQ(num_mismatches, 0) << name << " has mismatches";
    }

    // Helper: Create identical initial splat data
    std::pair<gs::SplatData, lfs::core::SplatData> createIdenticalSplatData(size_t N, int sh_degree = 3) {
        // Create legacy splat data
        auto means_legacy = torch::randn({static_cast<int64_t>(N), 3},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto sh0_legacy = torch::randn({static_cast<int64_t>(N), 1, 3},
                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        size_t sh_rest_coeffs = (sh_degree + 1) * (sh_degree + 1) - 1;
        auto shN_legacy = torch::randn({static_cast<int64_t>(N), static_cast<int64_t>(sh_rest_coeffs), 3},
                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto scaling_legacy = torch::randn({static_cast<int64_t>(N), 3},
                                          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto rotation_legacy = torch::randn({static_cast<int64_t>(N), 4},
                                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Normalize quaternions
        auto rot_norm = rotation_legacy.norm(2, -1, true);
        rotation_legacy = rotation_legacy / rot_norm;

        auto opacity_legacy = torch::randn({static_cast<int64_t>(N), 1},
                                          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Set requires_grad for legacy
        means_legacy.requires_grad_(true);
        sh0_legacy.requires_grad_(true);
        shN_legacy.requires_grad_(true);
        scaling_legacy.requires_grad_(true);
        rotation_legacy.requires_grad_(true);
        opacity_legacy.requires_grad_(true);

        // Create new splat data by copying from legacy
        auto copy_to_lfs = [](const torch::Tensor& t) -> Tensor {
            auto cpu = t.cpu().contiguous();
            const float* data = cpu.data_ptr<float>();

            std::vector<size_t> shape;
            for (int i = 0; i < cpu.dim(); ++i) {
                shape.push_back(cpu.size(i));
            }

            auto cpu_tensor = Tensor::from_blob(const_cast<float*>(data), TensorShape(shape),
                                               Device::CPU, DataType::Float32);
            return cpu_tensor.cuda();
        };

        auto means_new = copy_to_lfs(means_legacy);
        auto sh0_new = copy_to_lfs(sh0_legacy);
        auto shN_new = copy_to_lfs(shN_legacy);
        auto scaling_new = copy_to_lfs(scaling_legacy);
        auto rotation_new = copy_to_lfs(rotation_legacy);
        auto opacity_new = copy_to_lfs(opacity_legacy);

        gs::SplatData legacy_splat(sh_degree, means_legacy, sh0_legacy, shN_legacy,
                                    scaling_legacy, rotation_legacy, opacity_legacy, 1.0f);

        lfs::core::SplatData new_splat(sh_degree, std::move(means_new), std::move(sh0_new),
                                        std::move(shN_new), std::move(scaling_new),
                                        std::move(rotation_new), std::move(opacity_new), 1.0f);

        new_splat.allocate_gradients();

        return {std::move(legacy_splat), std::move(new_splat)};
    }

    // Helper: Create matching parameters
    std::pair<gs::param::OptimizationParameters, lfs::core::param::OptimizationParameters>
    createMatchingParams() {
        gs::param::OptimizationParameters legacy_params;
        legacy_params.iterations = 30000;
        legacy_params.means_lr = 1.6e-4f;
        legacy_params.min_opacity = 0.005f;
        legacy_params.max_cap = 1000000;
        legacy_params.start_refine = 500;
        legacy_params.stop_refine = 15000;
        legacy_params.refine_every = 100;
        legacy_params.sh_degree_interval = 1000;

        lfs::core::param::OptimizationParameters new_params;
        new_params.iterations = legacy_params.iterations;
        new_params.means_lr = legacy_params.means_lr;
        new_params.min_opacity = legacy_params.min_opacity;
        new_params.max_cap = legacy_params.max_cap;
        new_params.start_refine = legacy_params.start_refine;
        new_params.stop_refine = legacy_params.stop_refine;
        new_params.refine_every = legacy_params.refine_every;
        new_params.sh_degree_interval = legacy_params.sh_degree_interval;

        return {legacy_params, new_params};
    }

    // Helper: Initialize optimizer state by running a few steps with identical gradients
    void warmUpOptimizers(gs::training::MCMC& legacy_strategy,
                         lfs::training::MCMC& new_strategy,
                         int num_steps = 5) {
        for (int i = 0; i < num_steps; ++i) {
            // Set identical gradients
            auto& legacy_model = legacy_strategy.get_model();
            auto& new_model = new_strategy.get_model();

            // Allocate legacy gradients if needed
            if (!legacy_model.means().grad().defined()) {
                legacy_model.means().mutable_grad() = torch::zeros_like(legacy_model.means());
            }
            if (!legacy_model.sh0().grad().defined()) {
                legacy_model.sh0().mutable_grad() = torch::zeros_like(legacy_model.sh0());
            }
            if (!legacy_model.shN().grad().defined()) {
                legacy_model.shN().mutable_grad() = torch::zeros_like(legacy_model.shN());
            }
            if (!legacy_model.scaling_raw().grad().defined()) {
                legacy_model.scaling_raw().mutable_grad() = torch::zeros_like(legacy_model.scaling_raw());
            }
            if (!legacy_model.rotation_raw().grad().defined()) {
                legacy_model.rotation_raw().mutable_grad() = torch::zeros_like(legacy_model.rotation_raw());
            }
            if (!legacy_model.opacity_raw().grad().defined()) {
                legacy_model.opacity_raw().mutable_grad() = torch::zeros_like(legacy_model.opacity_raw());
            }

            legacy_model.means().mutable_grad().fill_(0.001f);
            legacy_model.sh0().mutable_grad().fill_(0.001f);
            legacy_model.shN().mutable_grad().fill_(0.0005f);
            legacy_model.scaling_raw().mutable_grad().fill_(0.001f);
            legacy_model.rotation_raw().mutable_grad().fill_(0.001f);
            legacy_model.opacity_raw().mutable_grad().fill_(0.001f);

            new_model.means_grad().fill_(0.001f);
            new_model.sh0_grad().fill_(0.001f);
            new_model.shN_grad().fill_(0.0005f);
            new_model.scaling_grad().fill_(0.001f);
            new_model.rotation_grad().fill_(0.001f);
            new_model.opacity_grad().fill_(0.001f);

            // Step both optimizers
            legacy_strategy.step(i);
            new_strategy.step(i);
        }
    }

    // Helper: Call add_new_gs with manually specified indices (bypassing multinomial)
    void addNewGsWithFixedIndices(gs::training::MCMC& legacy_strategy,
                                  lfs::training::MCMC& new_strategy,
                                  const std::vector<int32_t>& sample_indices) {
        spdlog::info("Calling add_new_gs with {} fixed indices", sample_indices.size());

        // Create index tensor for legacy
        auto legacy_indices = torch::tensor(sample_indices,
                                           torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        // Create index tensor for new
        auto new_indices = Tensor::from_vector(sample_indices, TensorShape({sample_indices.size()}),
                                              Device::CUDA);

        // Call the internal add_new_gs_with_indices function
        // We need to access the protected/private methods, so we'll use the public test interface
        int legacy_added = legacy_strategy.add_new_gs_with_indices_test(legacy_indices);
        int new_added = new_strategy.add_new_gs_with_indices_test(new_indices);

        spdlog::info("Legacy added: {}, New added: {}", legacy_added, new_added);

        EXPECT_EQ(legacy_added, new_added) << "Number of added Gaussians differs";
        EXPECT_EQ(static_cast<size_t>(legacy_added), sample_indices.size())
            << "Should add exactly the number of specified indices";
    }
};

// ============================================================================
// Single Iteration Tests
// ============================================================================

TEST_F(AddNewGsDeterministicTest, SingleIteration_IdenticalInitialState) {
    spdlog::info("=== Testing add_new_gs - Single Iteration, Identical Initial State ===");

    constexpr size_t N = 100;

    auto [legacy_splat, new_splat] = createIdenticalSplatData(N);
    auto [legacy_params, new_params] = createMatchingParams();

    // Verify initial states match
    spdlog::info("=== Verifying Initial States Match ===");
    compareTensors(legacy_splat.means(), new_splat.means(), "Initial Means");
    compareTensors(legacy_splat.opacity_raw(), new_splat.opacity_raw(), "Initial Opacity");
    compareTensors(legacy_splat.scaling_raw(), new_splat.scaling_raw(), "Initial Scaling");

    // Create strategies
    gs::training::MCMC legacy_strategy(std::move(legacy_splat));
    lfs::training::MCMC new_strategy(std::move(new_splat));

    legacy_strategy.initialize(legacy_params);
    new_strategy.initialize(new_params);

    // Warm up optimizers with identical gradients
    spdlog::info("=== Warming Up Optimizers ===");
    warmUpOptimizers(legacy_strategy, new_strategy, 5);

    // Verify states still match after warmup
    spdlog::info("=== Verifying States After Warmup ===");
    compareTensors(legacy_strategy.get_model().means(), new_strategy.get_model().means(),
                  "Means After Warmup");
    compareTensors(legacy_strategy.get_model().opacity_raw(), new_strategy.get_model().opacity_raw(),
                  "Opacity After Warmup");

    // Define fixed indices to sample (5% of N = 5 Gaussians)
    std::vector<int32_t> sample_indices = {10, 25, 50, 75, 90};

    size_t initial_count = legacy_strategy.get_model().size();

    // Call add_new_gs with fixed indices
    spdlog::info("=== Calling add_new_gs with Fixed Indices ===");
    addNewGsWithFixedIndices(legacy_strategy, new_strategy, sample_indices);

    size_t legacy_final = legacy_strategy.get_model().size();
    size_t new_final = new_strategy.get_model().size();

    spdlog::info("Initial: {}, Legacy final: {}, New final: {}", initial_count, legacy_final, new_final);

    EXPECT_EQ(legacy_final, new_final) << "Final Gaussian counts differ";
    EXPECT_EQ(legacy_final, initial_count + sample_indices.size())
        << "Should have exactly initial + sampled Gaussians";

    // Compare the original Gaussians (they should have updated opacity/scaling)
    spdlog::info("=== Comparing Original Gaussians (Updated) ===");
    auto legacy_means_full = legacy_strategy.get_model().means();
    auto new_means_full = new_strategy.get_model().means();

    auto legacy_means_orig = legacy_means_full.slice(0, 0, initial_count);
    auto new_means_orig = new_means_full.slice(0, 0, initial_count);

    compareTensors(legacy_means_orig, new_means_orig, "Original Means (after add_new_gs)");

    auto legacy_opacity_orig = legacy_strategy.get_model().get_opacity().slice(0, 0, initial_count);
    auto new_opacity_orig = new_strategy.get_model().get_opacity().slice(0, 0, initial_count);

    compareTensors(legacy_opacity_orig, new_opacity_orig, "Original Opacity (after add_new_gs)");

    auto legacy_scaling_orig = legacy_strategy.get_model().get_scaling().slice(0, 0, initial_count);
    auto new_scaling_orig = new_strategy.get_model().get_scaling().slice(0, 0, initial_count);

    compareTensors(legacy_scaling_orig, new_scaling_orig, "Original Scaling (after add_new_gs)");

    // Compare the newly added Gaussians (they should match the sampled indices)
    spdlog::info("=== Comparing Newly Added Gaussians ===");
    auto legacy_means_new = legacy_means_full.slice(0, initial_count, legacy_final);
    auto new_means_new = new_means_full.slice(0, initial_count, new_final);

    compareTensors(legacy_means_new, new_means_new, "New Gaussians Means");

    auto legacy_opacity_new = legacy_strategy.get_model().get_opacity().slice(0, initial_count, legacy_final);
    auto new_opacity_new = new_strategy.get_model().get_opacity().slice(0, initial_count, new_final);

    compareTensors(legacy_opacity_new, new_opacity_new, "New Gaussians Opacity");

    spdlog::info("=== Single Iteration Test Complete ===");
}

// ============================================================================
// Multiple Iteration Tests
// ============================================================================

TEST_F(AddNewGsDeterministicTest, MultipleIterations_ConsistentBehavior) {
    spdlog::info("=== Testing add_new_gs - Multiple Iterations ===");

    constexpr size_t N = 100;
    constexpr int num_iterations = 10;

    auto [legacy_splat, new_splat] = createIdenticalSplatData(N);
    auto [legacy_params, new_params] = createMatchingParams();

    gs::training::MCMC legacy_strategy(std::move(legacy_splat));
    lfs::training::MCMC new_strategy(std::move(new_splat));

    legacy_strategy.initialize(legacy_params);
    new_strategy.initialize(new_params);

    // Warm up optimizers
    warmUpOptimizers(legacy_strategy, new_strategy, 5);

    size_t current_count = N;

    for (int iter = 0; iter < num_iterations; ++iter) {
        spdlog::info("=== Iteration {} ===", iter);

        // Define different sampling indices for each iteration
        std::vector<int32_t> sample_indices;
        size_t num_to_sample = std::min<size_t>(5, current_count / 2);

        for (size_t i = 0; i < num_to_sample; ++i) {
            sample_indices.push_back(static_cast<int32_t>(i * current_count / num_to_sample));
        }

        spdlog::info("Sampling {} Gaussians from {} total", sample_indices.size(), current_count);

        // Add new Gaussians
        addNewGsWithFixedIndices(legacy_strategy, new_strategy, sample_indices);

        size_t legacy_size = legacy_strategy.get_model().size();
        size_t new_size = new_strategy.get_model().size();

        EXPECT_EQ(legacy_size, new_size) << "Sizes differ at iteration " << iter;
        EXPECT_EQ(legacy_size, current_count + sample_indices.size())
            << "Size not as expected at iteration " << iter;

        // Compare states after each iteration
        spdlog::info("Comparing states after iteration {}", iter);

        // Compare original Gaussians (first current_count)
        auto legacy_means_orig = legacy_strategy.get_model().means().slice(0, 0, current_count);
        auto new_means_orig = new_strategy.get_model().means().slice(0, 0, current_count);

        compareTensors(legacy_means_orig, new_means_orig,
                      "Original Means (iter " + std::to_string(iter) + ")");

        auto legacy_opacity_orig = legacy_strategy.get_model().get_opacity().slice(0, 0, current_count);
        auto new_opacity_orig = new_strategy.get_model().get_opacity().slice(0, 0, current_count);

        compareTensors(legacy_opacity_orig, new_opacity_orig,
                      "Original Opacity (iter " + std::to_string(iter) + ")");

        // Update current count for next iteration
        current_count = legacy_size;

        // Take optimizer steps with identical gradients
        auto& legacy_model = legacy_strategy.get_model();
        auto& new_model = new_strategy.get_model();

        legacy_model.means().mutable_grad().fill_(0.001f);
        legacy_model.sh0().mutable_grad().fill_(0.001f);
        legacy_model.shN().mutable_grad().fill_(0.0005f);
        legacy_model.scaling_raw().mutable_grad().fill_(0.001f);
        legacy_model.rotation_raw().mutable_grad().fill_(0.001f);
        legacy_model.opacity_raw().mutable_grad().fill_(0.001f);

        new_model.means_grad().fill_(0.001f);
        new_model.sh0_grad().fill_(0.001f);
        new_model.shN_grad().fill_(0.0005f);
        new_model.scaling_grad().fill_(0.001f);
        new_model.rotation_grad().fill_(0.001f);
        new_model.opacity_grad().fill_(0.001f);

        legacy_strategy.step(iter + 100);
        new_strategy.step(iter + 100);
    }

    spdlog::info("=== Multiple Iterations Test Complete ===");
    spdlog::info("Final size - Legacy: {}, New: {}",
                legacy_strategy.get_model().size(), new_strategy.get_model().size());
}

// ============================================================================
// Optimizer State Tests
// ============================================================================

TEST_F(AddNewGsDeterministicTest, OptimizerState_ConsistentAfterGrowth) {
    spdlog::info("=== Testing Optimizer State Consistency After Growth ===");

    constexpr size_t N = 100;

    auto [legacy_splat, new_splat] = createIdenticalSplatData(N);
    auto [legacy_params, new_params] = createMatchingParams();

    gs::training::MCMC legacy_strategy(std::move(legacy_splat));
    lfs::training::MCMC new_strategy(std::move(new_splat));

    legacy_strategy.initialize(legacy_params);
    new_strategy.initialize(new_params);

    // Warm up optimizers to build state
    warmUpOptimizers(legacy_strategy, new_strategy, 10);

    // Add new Gaussians
    std::vector<int32_t> sample_indices = {10, 20, 30, 40, 50};
    addNewGsWithFixedIndices(legacy_strategy, new_strategy, sample_indices);

    // Take more steps to use the new optimizer state
    spdlog::info("=== Taking Steps After Growth ===");
    for (int i = 0; i < 10; ++i) {
        auto& legacy_model = legacy_strategy.get_model();
        auto& new_model = new_strategy.get_model();

        // Allocate gradients if they were reset by zero_grad(set_to_none=true)
        if (!legacy_model.means().grad().defined()) {
            legacy_model.means().mutable_grad() = torch::zeros_like(legacy_model.means());
        }
        if (!legacy_model.sh0().grad().defined()) {
            legacy_model.sh0().mutable_grad() = torch::zeros_like(legacy_model.sh0());
        }
        if (!legacy_model.shN().grad().defined()) {
            legacy_model.shN().mutable_grad() = torch::zeros_like(legacy_model.shN());
        }
        if (!legacy_model.scaling_raw().grad().defined()) {
            legacy_model.scaling_raw().mutable_grad() = torch::zeros_like(legacy_model.scaling_raw());
        }
        if (!legacy_model.rotation_raw().grad().defined()) {
            legacy_model.rotation_raw().mutable_grad() = torch::zeros_like(legacy_model.rotation_raw());
        }
        if (!legacy_model.opacity_raw().grad().defined()) {
            legacy_model.opacity_raw().mutable_grad() = torch::zeros_like(legacy_model.opacity_raw());
        }

        legacy_model.means().mutable_grad().fill_(0.002f);
        legacy_model.sh0().mutable_grad().fill_(0.002f);
        legacy_model.shN().mutable_grad().fill_(0.001f);
        legacy_model.scaling_raw().mutable_grad().fill_(0.002f);
        legacy_model.rotation_raw().mutable_grad().fill_(0.002f);
        legacy_model.opacity_raw().mutable_grad().fill_(0.002f);

        new_model.means_grad().fill_(0.002f);
        new_model.sh0_grad().fill_(0.002f);
        new_model.shN_grad().fill_(0.001f);
        new_model.scaling_grad().fill_(0.002f);
        new_model.rotation_grad().fill_(0.002f);
        new_model.opacity_grad().fill_(0.002f);

        legacy_strategy.step(100 + i);
        new_strategy.step(100 + i);

        if (i % 5 == 0) {
            spdlog::info("Comparing after step {}", i);
            compareTensors(legacy_strategy.get_model().means(),
                          new_strategy.get_model().means(),
                          "Means (step " + std::to_string(i) + " after growth)",
                          1e-4f);  // Relaxed tolerance due to accumulated updates
        }
    }

    spdlog::info("=== Optimizer State Test Complete ===");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(AddNewGsDeterministicTest, EdgeCase_SampleAllGaussians) {
    spdlog::info("=== Testing Edge Case - Sample All Gaussians ===");

    constexpr size_t N = 50;

    auto [legacy_splat, new_splat] = createIdenticalSplatData(N);
    auto [legacy_params, new_params] = createMatchingParams();

    gs::training::MCMC legacy_strategy(std::move(legacy_splat));
    lfs::training::MCMC new_strategy(std::move(new_splat));

    legacy_strategy.initialize(legacy_params);
    new_strategy.initialize(new_params);

    warmUpOptimizers(legacy_strategy, new_strategy, 5);

    // Sample ALL Gaussians
    std::vector<int32_t> sample_indices;
    for (size_t i = 0; i < N; ++i) {
        sample_indices.push_back(static_cast<int32_t>(i));
    }

    spdlog::info("Sampling all {} Gaussians", sample_indices.size());
    addNewGsWithFixedIndices(legacy_strategy, new_strategy, sample_indices);

    EXPECT_EQ(legacy_strategy.get_model().size(), N * 2);
    EXPECT_EQ(new_strategy.get_model().size(), N * 2);

    compareTensors(legacy_strategy.get_model().means(),
                  new_strategy.get_model().means(),
                  "Means (after sampling all)");

    spdlog::info("=== Edge Case Test Complete ===");
}

TEST_F(AddNewGsDeterministicTest, EdgeCase_SampleSingleGaussian) {
    spdlog::info("=== Testing Edge Case - Sample Single Gaussian ===");

    constexpr size_t N = 100;

    auto [legacy_splat, new_splat] = createIdenticalSplatData(N);
    auto [legacy_params, new_params] = createMatchingParams();

    gs::training::MCMC legacy_strategy(std::move(legacy_splat));
    lfs::training::MCMC new_strategy(std::move(new_splat));

    legacy_strategy.initialize(legacy_params);
    new_strategy.initialize(new_params);

    warmUpOptimizers(legacy_strategy, new_strategy, 5);

    // Sample just one Gaussian
    std::vector<int32_t> sample_indices = {42};

    spdlog::info("Sampling single Gaussian at index 42");
    addNewGsWithFixedIndices(legacy_strategy, new_strategy, sample_indices);

    EXPECT_EQ(legacy_strategy.get_model().size(), N + 1);
    EXPECT_EQ(new_strategy.get_model().size(), N + 1);

    // The new Gaussian should be a copy of Gaussian 42
    auto legacy_means = legacy_strategy.get_model().means();
    auto new_means = new_strategy.get_model().means();

    // Compare the newly added Gaussian (last one) with the original (index 42)
    auto legacy_original_42 = legacy_means.slice(0, 42, 43);
    auto legacy_new = legacy_means.slice(0, N, N + 1);

    auto new_original_42 = new_means.slice(0, 42, 43);
    auto new_new = new_means.slice(0, N, N + 1);

    // New Gaussian means should match original (before relocation)
    compareTensors(legacy_new, new_new, "Newly Added Gaussian Means");

    spdlog::info("=== Edge Case Test Complete ===");
}

TEST_F(AddNewGsDeterministicTest, EdgeCase_RepeatedIndices) {
    spdlog::info("=== Testing Edge Case - Repeated Indices ===");

    constexpr size_t N = 100;

    auto [legacy_splat, new_splat] = createIdenticalSplatData(N);
    auto [legacy_params, new_params] = createMatchingParams();

    gs::training::MCMC legacy_strategy(std::move(legacy_splat));
    lfs::training::MCMC new_strategy(std::move(new_splat));

    legacy_strategy.initialize(legacy_params);
    new_strategy.initialize(new_params);

    warmUpOptimizers(legacy_strategy, new_strategy, 5);

    // Sample the same Gaussian multiple times
    std::vector<int32_t> sample_indices = {10, 10, 10, 20, 20};

    spdlog::info("Sampling with repeated indices: [10, 10, 10, 20, 20]");
    addNewGsWithFixedIndices(legacy_strategy, new_strategy, sample_indices);

    EXPECT_EQ(legacy_strategy.get_model().size(), N + 5);
    EXPECT_EQ(new_strategy.get_model().size(), N + 5);

    compareTensors(legacy_strategy.get_model().means(),
                  new_strategy.get_model().means(),
                  "Means (after repeated sampling)");

    spdlog::info("=== Edge Case Test Complete ===");
}
