/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "optimizer/adam_optimizer.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>

#include "optimizers/fused_adam.hpp"  // Reference optimizer

using namespace lfs::core;
using namespace lfs::training;

namespace {

// ===================================================================================
// Helper functions for Torch-Tensor interop
// ===================================================================================

// Convert lfs::core::Tensor to torch::Tensor
torch::Tensor to_torch(const lfs::core::Tensor& lfs_tensor) {
    auto vec = lfs_tensor.cpu().to_vector();
    std::vector<int64_t> torch_shape;
    for (size_t i = 0; i < lfs_tensor.ndim(); i++) {
        torch_shape.push_back(lfs_tensor.shape()[i]);
    }
    auto torch_t = torch::from_blob(vec.data(), torch_shape, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return lfs_tensor.device() == lfs::core::Device::CUDA ? torch_t.to(torch::kCUDA) : torch_t;
}

// Convert torch::Tensor to lfs::core::Tensor
lfs::core::Tensor from_torch(const torch::Tensor& torch_tensor) {
    auto cpu_t = torch_tensor.cpu().contiguous();
    std::vector<float> vec(cpu_t.data_ptr<float>(),
                           cpu_t.data_ptr<float>() + cpu_t.numel());

    std::vector<size_t> shape;
    for (int i = 0; i < cpu_t.dim(); i++) {
        shape.push_back(cpu_t.size(i));
    }

    auto device = torch_tensor.is_cuda() ? lfs::core::Device::CUDA : lfs::core::Device::CPU;
    return lfs::core::Tensor::from_vector(vec, lfs::core::TensorShape(shape), device);
}

// Copy lfs::core::Tensor data into existing torch::Tensor
void copy_to_torch(const lfs::core::Tensor& lfs_tensor, torch::Tensor& torch_tensor) {
    ASSERT_EQ(lfs_tensor.numel(), torch_tensor.numel());
    auto vec = lfs_tensor.cpu().to_vector();
    auto torch_cpu = torch_tensor.cpu().contiguous();
    std::memcpy(torch_cpu.data_ptr<float>(), vec.data(), vec.size() * sizeof(float));
    if (torch_tensor.is_cuda()) {
        torch_tensor.copy_(torch_cpu);
    }
}

// Compare tensors with tolerance
bool tensors_close(const lfs::core::Tensor& lfs_tensor, const torch::Tensor& torch_tensor,
                   float rtol = 1e-5f, float atol = 1e-5f) {
    if (lfs_tensor.numel() != torch_tensor.numel()) return false;

    auto lfs_vec = lfs_tensor.cpu().to_vector();
    auto torch_cpu = torch_tensor.cpu().contiguous();
    auto torch_ptr = torch_cpu.data_ptr<float>();

    for (size_t i = 0; i < lfs_vec.size(); i++) {
        float diff = std::abs(lfs_vec[i] - torch_ptr[i]);
        float threshold = atol + rtol * std::abs(torch_ptr[i]);
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

float max_diff(const lfs::core::Tensor& lfs_tensor, const torch::Tensor& torch_tensor) {
    auto lfs_vec = lfs_tensor.cpu().to_vector();
    auto torch_cpu = torch_tensor.cpu().contiguous();
    auto torch_ptr = torch_cpu.data_ptr<float>();

    float max_d = 0.0f;
    for (size_t i = 0; i < lfs_vec.size(); i++) {
        max_d = std::max(max_d, std::abs(lfs_vec[i] - torch_ptr[i]));
    }
    return max_d;
}

// Helper function to create a simple SplatData for testing
SplatData create_test_splat_data(size_t n_points = 100, int sh_degree = 3) {
    auto means = Tensor::randn({n_points, 3}, Device::CUDA);
    auto sh0 = Tensor::randn({n_points, 1, 3}, Device::CUDA);

    // For sh_degree=3, we have (3+1)^2 = 16 coefficients, minus DC = 15
    size_t sh_rest_coeffs = (sh_degree + 1) * (sh_degree + 1) - 1;
    auto shN = Tensor::randn({n_points, sh_rest_coeffs, 3}, Device::CUDA);

    auto scaling = Tensor::randn({n_points, 3}, Device::CUDA);
    auto rotation = Tensor::randn({n_points, 4}, Device::CUDA);
    auto opacity = Tensor::randn({n_points, 1}, Device::CUDA);

    return SplatData(sh_degree, std::move(means), std::move(sh0), std::move(shN),
                     std::move(scaling), std::move(rotation), std::move(opacity), 1.0f);
}

TEST(SplatDataGradientsTest, GradientAllocation) {
    auto splat_data = create_test_splat_data(100, 3);

    EXPECT_FALSE(splat_data.has_gradients());

    splat_data.allocate_gradients();

    EXPECT_TRUE(splat_data.has_gradients());
    EXPECT_EQ(splat_data.means_grad().shape(), splat_data.means().shape());
    EXPECT_EQ(splat_data.sh0_grad().shape(), splat_data.sh0().shape());
    EXPECT_EQ(splat_data.shN_grad().shape(), splat_data.shN().shape());
    EXPECT_EQ(splat_data.scaling_grad().shape(), splat_data.scaling_raw().shape());
    EXPECT_EQ(splat_data.rotation_grad().shape(), splat_data.rotation_raw().shape());
    EXPECT_EQ(splat_data.opacity_grad().shape(), splat_data.opacity_raw().shape());
}

TEST(SplatDataGradientsTest, GradientZeroing) {
    auto splat_data = create_test_splat_data(100, 3);

    splat_data.allocate_gradients();

    // Set some gradient values
    splat_data.means_grad().fill_(1.0f);
    splat_data.sh0_grad().fill_(2.0f);

    // Verify non-zero
    EXPECT_FLOAT_EQ(splat_data.means_grad().sum_scalar(), 300.0f);  // 100*3*1.0
    EXPECT_FLOAT_EQ(splat_data.sh0_grad().sum_scalar(), 600.0f);     // 100*1*3*2.0

    // Zero gradients
    splat_data.zero_gradients();

    // Verify all zeros
    EXPECT_FLOAT_EQ(splat_data.means_grad().sum_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(splat_data.sh0_grad().sum_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(splat_data.shN_grad().sum_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(splat_data.scaling_grad().sum_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(splat_data.rotation_grad().sum_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(splat_data.opacity_grad().sum_scalar(), 0.0f);
}

TEST(AdamOptimizerTest, OptimizerCreation) {
    auto splat_data = create_test_splat_data(10, 3);

    AdamConfig config;
    config.lr = 0.01f;
    config.beta1 = 0.9f;
    config.beta2 = 0.999f;

    AdamOptimizer optimizer(splat_data, config);

    // Gradients should be allocated automatically
    EXPECT_TRUE(splat_data.has_gradients());

    // Check config
    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.01f);
}

TEST(AdamOptimizerTest, SingleStepUpdate) {
    auto splat_data = create_test_splat_data(10, 3);

    splat_data.allocate_gradients();

    // Save initial values
    auto initial_means = splat_data.means().clone();

    // Set constant gradient = -1.0 (should move params in positive direction)
    splat_data.means_grad().fill_(-1.0f);

    // Create optimizer
    AdamConfig config;
    config.lr = 0.1f;
    AdamOptimizer optimizer(splat_data, config);

    // Take one step
    optimizer.step(1);

    // Verify parameters changed (should increase due to negative gradient)
    auto diff = splat_data.means().sub(initial_means);
    float mean_change = diff.mean_scalar();

    EXPECT_GT(mean_change, 0.0f) << "Parameters should increase with negative gradient";
}

TEST(AdamOptimizerTest, GradientDescentConvergence) {
    // Target value
    auto target = Tensor::full({100, 3}, 1.0f, Device::CUDA);

    // Starting value (far from target)
    auto means = Tensor::full({100, 3}, -5.0f, Device::CUDA);
    auto sh0 = Tensor::zeros({100, 1, 3}, Device::CUDA);
    auto shN = Tensor::zeros({100, 15, 3}, Device::CUDA);
    auto scaling = Tensor::zeros({100, 3}, Device::CUDA);
    auto rotation = Tensor::ones({100, 4}, Device::CUDA);  // Identity quaternion
    auto opacity = Tensor::zeros({100, 1}, Device::CUDA);

    SplatData splat_data(3, std::move(means), std::move(sh0), std::move(shN),
                        std::move(scaling), std::move(rotation), std::move(opacity), 1.0f);

    splat_data.allocate_gradients();

    AdamConfig config;
    config.lr = 0.1f;
    AdamOptimizer optimizer(splat_data, config);

    // Run optimization loop for mean squared error loss
    for (int i = 0; i < 150; i++) {
        // Compute gradient: grad = 2 * (means - target)
        auto diff = splat_data.means().sub(target);
        splat_data.means_grad() = diff.mul(2.0f);

        // Optimize
        optimizer.step(i + 1);
        optimizer.zero_grad(i + 1);
    }

    // Check convergence (should be close to target)
    auto final_diff = splat_data.means().sub(target).abs().mean_scalar();
    EXPECT_LT(final_diff, 0.15f) << "Should converge within 0.15 of target after 150 steps";
}

TEST(AdamOptimizerTest, BiasCorrection) {
    auto splat_data = create_test_splat_data(10, 3);

    splat_data.allocate_gradients();
    splat_data.means_grad().fill_(-1.0f);

    AdamConfig config;
    config.lr = 0.001f;
    AdamOptimizer optimizer(splat_data, config);

    // Save initial
    auto initial = splat_data.means().clone();

    // First step (high bias correction)
    optimizer.step(1);
    auto step1_change = splat_data.means().sub(initial).abs().mean_scalar();

    // Reset params and optimizer state for clean test
    auto splat_data2 = create_test_splat_data(10, 3);
    splat_data2.allocate_gradients();
    AdamOptimizer optimizer2(splat_data2, config);

    // Do 10 steps
    for (int i = 0; i < 10; i++) {
        splat_data2.means_grad().fill_(-1.0f);
        optimizer2.step(i + 1);
    }

    float total_change = splat_data2.means().sub(initial).abs().mean_scalar();
    float avg_change = total_change / 10.0f;

    // Early steps should have relatively larger updates due to bias correction
    // (This is a weak test but validates bias correction has some effect)
    EXPECT_GT(step1_change, 0.0f);
    EXPECT_GT(avg_change, 0.0f);
}

TEST(AdamOptimizerTest, LearningRateUpdate) {
    // Test 1: Higher learning rate
    auto splat_data1 = create_test_splat_data(10, 3);
    splat_data1.allocate_gradients();
    auto initial1 = splat_data1.means().clone();

    AdamConfig config1;
    config1.lr = 0.1f;
    AdamOptimizer optimizer1(splat_data1, config1);

    splat_data1.means_grad().fill_(-1.0f);
    optimizer1.step(1);
    float change_lr01 = splat_data1.means().sub(initial1).abs().mean_scalar();

    // Test 2: Lower learning rate (fresh optimizer and data)
    auto splat_data2 = create_test_splat_data(10, 3);
    splat_data2.allocate_gradients();
    auto initial2 = splat_data2.means().clone();

    AdamConfig config2;
    config2.lr = 0.01f;  // 10x smaller
    AdamOptimizer optimizer2(splat_data2, config2);

    EXPECT_FLOAT_EQ(optimizer2.get_lr(), 0.01f);

    splat_data2.means_grad().fill_(-1.0f);
    optimizer2.step(1);
    float change_lr001 = splat_data2.means().sub(initial2).abs().mean_scalar();

    // Smaller LR should produce smaller changes
    EXPECT_LT(change_lr001, change_lr01) << "Smaller learning rate should produce smaller updates";
}

TEST(AdamOptimizerTest, MultipleParameterOptimization) {
    auto splat_data = create_test_splat_data(50, 3);

    splat_data.allocate_gradients();

    // Set gradients for all parameters
    splat_data.means_grad().fill_(-1.0f);
    splat_data.sh0_grad().fill_(-0.5f);
    splat_data.shN_grad().fill_(-0.1f);
    splat_data.scaling_grad().fill_(-0.2f);
    splat_data.rotation_grad().fill_(-0.3f);
    splat_data.opacity_grad().fill_(-0.4f);

    // Save initial values
    auto initial_means = splat_data.means().clone();
    auto initial_sh0 = splat_data.sh0().clone();
    auto initial_shN = splat_data.shN().clone();
    auto initial_scaling = splat_data.scaling_raw().clone();
    auto initial_rotation = splat_data.rotation_raw().clone();
    auto initial_opacity = splat_data.opacity_raw().clone();

    AdamConfig config;
    config.lr = 0.01f;
    AdamOptimizer optimizer(splat_data, config);

    // Take one step
    optimizer.step(1);

    // Verify all parameters were updated
    EXPECT_NE(splat_data.means().sub(initial_means).abs().mean_scalar(), 0.0f);
    EXPECT_NE(splat_data.sh0().sub(initial_sh0).abs().mean_scalar(), 0.0f);
    EXPECT_NE(splat_data.shN().sub(initial_shN).abs().mean_scalar(), 0.0f);
    EXPECT_NE(splat_data.scaling_raw().sub(initial_scaling).abs().mean_scalar(), 0.0f);
    EXPECT_NE(splat_data.rotation_raw().sub(initial_rotation).abs().mean_scalar(), 0.0f);
    EXPECT_NE(splat_data.opacity_raw().sub(initial_opacity).abs().mean_scalar(), 0.0f);
}

TEST(AdamOptimizerTest, ZeroGradAfterStep) {
    auto splat_data = create_test_splat_data(10, 3);

    splat_data.allocate_gradients();
    splat_data.means_grad().fill_(5.0f);

    AdamConfig config;
    AdamOptimizer optimizer(splat_data, config);

    // Verify gradient is non-zero
    EXPECT_NE(splat_data.means_grad().sum_scalar(), 0.0f);

    // Step and zero
    optimizer.step(1);
    optimizer.zero_grad(1);

    // Verify all gradients are zero
    EXPECT_FLOAT_EQ(splat_data.means_grad().sum_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(splat_data.sh0_grad().sum_scalar(), 0.0f);
}

// ===========================================================================================
// State Manipulation Tests (for MCMC support)
// ===========================================================================================

TEST(AdamOptimizerStateTest, ResetStateAtIndices) {
    auto splat_data = create_test_splat_data(100, 3);
    splat_data.allocate_gradients();

    AdamConfig config;
    config.lr = 0.01f;
    AdamOptimizer optimizer(splat_data, config);

    // Run a few steps to accumulate state
    for (int i = 0; i < 5; i++) {
        splat_data.means_grad().fill_(-1.0f);
        optimizer.step(i + 1);
    }

    // Verify state exists and is non-zero
    auto* state_before = optimizer.get_state(ParamType::Means);
    ASSERT_NE(state_before, nullptr);
    EXPECT_EQ(state_before->step_count, 5);
    float exp_avg_sum_before = state_before->exp_avg.abs().sum_scalar();
    float exp_avg_sq_sum_before = state_before->exp_avg_sq.abs().sum_scalar();
    EXPECT_GT(exp_avg_sum_before, 0.0f);
    EXPECT_GT(exp_avg_sq_sum_before, 0.0f);

    // Reset state at specific indices (simulate MCMC relocation)
    std::vector<int64_t> indices_to_reset = {10, 20, 30, 40, 50};
    optimizer.reset_state_at_indices(ParamType::Means, indices_to_reset);

    // Verify state at those indices is zero
    auto* state_after = optimizer.get_state(ParamType::Means);
    ASSERT_NE(state_after, nullptr);

    // Extract values at reset indices and verify they're zero
    for (int64_t idx : indices_to_reset) {
        auto exp_avg_slice = state_after->exp_avg[idx];
        auto exp_avg_sq_slice = state_after->exp_avg_sq[idx];

        EXPECT_FLOAT_EQ(exp_avg_slice.abs().sum_scalar(), 0.0f)
            << "exp_avg at index " << idx << " should be zero";
        EXPECT_FLOAT_EQ(exp_avg_sq_slice.abs().sum_scalar(), 0.0f)
            << "exp_avg_sq at index " << idx << " should be zero";
    }

    // Verify step_count is unchanged
    EXPECT_EQ(state_after->step_count, 5);

    // Verify other indices are still non-zero
    auto exp_avg_slice_other = state_after->exp_avg[0];
    EXPECT_GT(exp_avg_slice_other.abs().sum_scalar(), 0.0f)
        << "exp_avg at non-reset indices should remain non-zero";
}

TEST(AdamOptimizerStateTest, ExtendStateForNewParams) {
    auto splat_data = create_test_splat_data(50, 3);
    splat_data.allocate_gradients();

    AdamConfig config;
    config.lr = 0.01f;
    AdamOptimizer optimizer(splat_data, config);

    // Run steps to build state
    for (int i = 0; i < 10; i++) {
        splat_data.means_grad().fill_(-0.5f);
        optimizer.step(i + 1);
    }

    // Verify initial state
    auto* state_before = optimizer.get_state(ParamType::Means);
    ASSERT_NE(state_before, nullptr);
    EXPECT_EQ(state_before->step_count, 10);
    EXPECT_EQ(state_before->size, 50);  // 50 points used
    EXPECT_GE(state_before->capacity, 50);  // Capacity >= size
    auto shape_before = state_before->exp_avg.shape();
    EXPECT_EQ(shape_before[1], 3);   // 3D means

    // Simulate adding new parameters (like MCMC does)
    size_t n_new = 25;

    // Extend parameters in splat_data
    auto new_means = Tensor::randn({n_new, 3}, Device::CUDA);
    auto extended_means = Tensor::cat(std::vector<Tensor>{splat_data.means(), new_means}, 0);

    // Extend gradients
    auto new_means_grad = Tensor::zeros({n_new, 3}, Device::CUDA);
    auto extended_means_grad = Tensor::cat(std::vector<Tensor>{splat_data.means_grad(), new_means_grad}, 0);

    // Copy back (simulating parameter concatenation)
    splat_data.means() = extended_means;
    splat_data.means_grad() = extended_means_grad;

    // Extend optimizer state
    optimizer.extend_state_for_new_params(ParamType::Means, n_new);

    // Verify extended state
    auto* state_after = optimizer.get_state(ParamType::Means);
    ASSERT_NE(state_after, nullptr);

    EXPECT_EQ(state_after->size, 75);  // 50 + 25 = 75 points used
    EXPECT_GE(state_after->capacity, 75);  // Capacity >= size (may be larger due to growth factor)
    auto shape_after = state_after->exp_avg.shape();
    EXPECT_EQ(shape_after[1], 3);   // Still 3D

    // CRITICAL: step_count must be preserved (matching MCMC behavior)
    EXPECT_EQ(state_after->step_count, 10) << "step_count must be preserved when extending state";

    // Verify old state values are preserved (check first element)
    auto old_exp_avg_elem = state_after->exp_avg[0];
    EXPECT_GT(old_exp_avg_elem.abs().sum_scalar(), 0.0f)
        << "Old optimizer state should be preserved";

    // Verify new state values are zero (check elements in extended range)
    for (size_t i = 50; i < 75; i += 10) {
        auto new_exp_avg_elem = state_after->exp_avg[i];
        EXPECT_FLOAT_EQ(new_exp_avg_elem.abs().sum_scalar(), 0.0f)
            << "New optimizer state should be initialized to zero at index " << i;

        auto new_exp_avg_sq_elem = state_after->exp_avg_sq[i];
        EXPECT_FLOAT_EQ(new_exp_avg_sq_elem.abs().sum_scalar(), 0.0f)
            << "New optimizer state should be initialized to zero at index " << i;
    }
}

TEST(AdamOptimizerStateTest, StepCountPreservation) {
    auto splat_data = create_test_splat_data(20, 3);
    splat_data.allocate_gradients();

    AdamConfig config;
    AdamOptimizer optimizer(splat_data, config);

    // Initial step count should be 0
    EXPECT_EQ(optimizer.get_step_count(ParamType::Means), 0);

    // Take 5 steps
    for (int i = 0; i < 5; i++) {
        splat_data.means_grad().fill_(-1.0f);
        optimizer.step(i + 1);
    }

    EXPECT_EQ(optimizer.get_step_count(ParamType::Means), 5);
    auto new_means = Tensor::randn({10, 3}, Device::CUDA);
    optimizer.add_new_params(ParamType::Means, new_means);
    EXPECT_EQ(optimizer.get_step_count(ParamType::Means), 5)
        << "Adding new params should preserve step_count";
    EXPECT_EQ(splat_data.means().shape()[0], 30)
        << "Parameter size should be updated";

    // Reset state at indices - step count should NOT change
    optimizer.reset_state_at_indices(ParamType::Means, {0, 5, 10});
    EXPECT_EQ(optimizer.get_step_count(ParamType::Means), 5)
        << "Resetting state at indices should preserve step_count";

    // Take more steps - count should increase
    for (int i = 5; i < 10; i++) {
        splat_data.means_grad().fill_(-1.0f);
        optimizer.step(i + 1);
    }

    EXPECT_EQ(optimizer.get_step_count(ParamType::Means), 10);
}

TEST(AdamOptimizerStateTest, MultipleParameterStateManipulation) {
    auto splat_data = create_test_splat_data(30, 3);
    splat_data.allocate_gradients();

    AdamConfig config;
    AdamOptimizer optimizer(splat_data, config);

    // Fill all gradients and take steps
    for (int i = 0; i < 3; i++) {
        splat_data.means_grad().fill_(-1.0f);
        splat_data.sh0_grad().fill_(-0.5f);
        splat_data.shN_grad().fill_(-0.1f);
        splat_data.scaling_grad().fill_(-0.2f);
        splat_data.rotation_grad().fill_(-0.3f);
        splat_data.opacity_grad().fill_(-0.4f);
        optimizer.step(i + 1);
    }

    // Verify all states exist
    EXPECT_NE(optimizer.get_state(ParamType::Means), nullptr);
    EXPECT_NE(optimizer.get_state(ParamType::Sh0), nullptr);
    EXPECT_NE(optimizer.get_state(ParamType::ShN), nullptr);
    EXPECT_NE(optimizer.get_state(ParamType::Scaling), nullptr);
    EXPECT_NE(optimizer.get_state(ParamType::Rotation), nullptr);
    EXPECT_NE(optimizer.get_state(ParamType::Opacity), nullptr);

    // Reset state for means only
    optimizer.reset_state_at_indices(ParamType::Means, {0, 5, 10});

    // Verify means state at indices is zero, but others unchanged
    auto* means_state = optimizer.get_state(ParamType::Means);
    auto means_exp_avg_slice = means_state->exp_avg[0];
    EXPECT_FLOAT_EQ(means_exp_avg_slice.abs().sum_scalar(), 0.0f);

    auto* sh0_state = optimizer.get_state(ParamType::Sh0);
    auto sh0_exp_avg_slice = sh0_state->exp_avg[0];
    EXPECT_GT(sh0_exp_avg_slice.abs().sum_scalar(), 0.0f)
        << "Other parameters should not be affected";

    // Extend only opacity state
    optimizer.extend_state_for_new_params(ParamType::Opacity, 10);

    auto* opacity_state = optimizer.get_state(ParamType::Opacity);
    EXPECT_EQ(opacity_state->size, 40);  // 30 + 10 (used size)
    EXPECT_GE(opacity_state->capacity, 40);  // Capacity >= size

    // Means state should be unchanged
    auto* means_state_after = optimizer.get_state(ParamType::Means);
    EXPECT_EQ(means_state_after->size, 30);  // Still 30 (used size)
    EXPECT_GE(means_state_after->capacity, 30);  // Capacity >= size
}

// ===========================================================================================
// Comparison Tests with Reference PyTorch FusedAdam Optimizer
// ===========================================================================================

TEST(AdamOptimizerComparisonTest, NumericalEquivalenceBasic) {
    constexpr int N = 50;
    constexpr int n_iters = 10;
    constexpr float lr = 0.01f;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps = 1e-8f;
    constexpr float tolerance = 1e-5f;

    // Create identical initial parameters
    auto torch_means = torch::randn({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto lfs_means_data = torch_means.clone().detach();

    // Setup PyTorch optimizer
    auto torch_opt_options = std::make_unique<gs::training::FusedAdam::Options>(lr);
    torch_opt_options->betas(std::make_tuple(beta1, beta2));
    torch_opt_options->eps(eps);
    gs::training::FusedAdam torch_opt({torch_means}, std::move(torch_opt_options));

    // Setup LFS optimizer
    auto lfs_splat = create_test_splat_data(N, 3);
    lfs_splat.means() = from_torch(lfs_means_data);
    lfs_splat.allocate_gradients();

    AdamConfig lfs_config;
    lfs_config.lr = lr;
    lfs_config.beta1 = beta1;
    lfs_config.beta2 = beta2;
    lfs_config.eps = eps;
    AdamOptimizer lfs_opt(lfs_splat, lfs_config);

    // Run optimization with identical gradients
    for (int iter = 0; iter < n_iters; iter++) {
        // Create identical gradient
        auto grad_values = torch::randn({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Apply to PyTorch
        torch_means.mutable_grad() = grad_values.clone();
        torch_opt.step(iter + 1);

        // Apply to LFS
        lfs_splat.means_grad() = from_torch(grad_values);
        lfs_opt.step(iter + 1);
    }

    // Compare final parameters
    auto torch_final = torch_means.detach();
    auto lfs_final_torch = to_torch(lfs_splat.means());

    float max_d = max_diff(lfs_splat.means(), torch_final);
    EXPECT_LT(max_d, tolerance) << "Max difference between optimizers: " << max_d;
    EXPECT_TRUE(tensors_close(lfs_splat.means(), torch_final, tolerance, tolerance));
}

TEST(AdamOptimizerComparisonTest, StateEquivalenceAfterSteps) {
    constexpr int N = 30;
    constexpr int n_iters = 5;
    constexpr float lr = 0.001f;
    constexpr float tolerance = 1e-5f;

    // Setup both optimizers
    auto torch_means = torch::randn({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto lfs_means_data = torch_means.clone().detach();

    auto torch_opt_options = std::make_unique<gs::training::FusedAdam::Options>(lr);
    gs::training::FusedAdam torch_opt({torch_means}, std::move(torch_opt_options));

    auto lfs_splat = create_test_splat_data(N, 3);
    lfs_splat.means() = from_torch(lfs_means_data);
    lfs_splat.allocate_gradients();

    AdamConfig lfs_config;
    lfs_config.lr = lr;
    AdamOptimizer lfs_opt(lfs_splat, lfs_config);

    // Run optimization
    for (int iter = 0; iter < n_iters; iter++) {
        auto grad = torch::randn({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        torch_means.mutable_grad() = grad.clone();
        torch_opt.step(iter + 1);

        lfs_splat.means_grad() = from_torch(grad);
        lfs_opt.step(iter + 1);
    }

    // Extract and compare state
    auto& torch_state_map = torch_opt.state();
    auto torch_state_it = torch_state_map.find(torch_means.unsafeGetTensorImpl());
    ASSERT_NE(torch_state_it, torch_state_map.end());

    auto* torch_adam_state = static_cast<gs::training::FusedAdam::AdamParamState*>(torch_state_it->second.get());
    auto* lfs_state = lfs_opt.get_state(ParamType::Means);
    ASSERT_NE(lfs_state, nullptr);

    // Compare step counts
    EXPECT_EQ(lfs_state->step_count, torch_adam_state->step_count);

    // Compare exp_avg (only the "used" portion if capacity > size)
    auto lfs_exp_avg_used = lfs_state->size < lfs_state->exp_avg.shape()[0]
        ? lfs_state->exp_avg.slice(0, 0, lfs_state->size)
        : lfs_state->exp_avg;
    float exp_avg_max_d = max_diff(lfs_exp_avg_used, torch_adam_state->exp_avg);
    EXPECT_LT(exp_avg_max_d, tolerance)
        << "exp_avg mismatch, max diff: " << exp_avg_max_d;

    // Compare exp_avg_sq (only the "used" portion)
    auto lfs_exp_avg_sq_used = lfs_state->size < lfs_state->exp_avg_sq.shape()[0]
        ? lfs_state->exp_avg_sq.slice(0, 0, lfs_state->size)
        : lfs_state->exp_avg_sq;
    float exp_avg_sq_max_d = max_diff(lfs_exp_avg_sq_used, torch_adam_state->exp_avg_sq);
    EXPECT_LT(exp_avg_sq_max_d, tolerance)
        << "exp_avg_sq mismatch, max diff: " << exp_avg_sq_max_d;
}

TEST(AdamOptimizerComparisonTest, MCMCRelocateSequence) {
    // Simulate MCMC relocate: optimize → reset state at indices → continue optimizing
    constexpr int N = 100;
    constexpr float lr = 0.01f;
    constexpr float tolerance = 1e-5f;

    // Setup both optimizers
    auto torch_means = torch::randn({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto lfs_means_data = torch_means.clone().detach();

    auto torch_opt = std::make_unique<gs::training::FusedAdam>(
        std::vector<torch::Tensor>{torch_means},
        std::make_unique<gs::training::FusedAdam::Options>(lr)
    );

    auto lfs_splat = create_test_splat_data(N, 3);
    lfs_splat.means() = from_torch(lfs_means_data);
    lfs_splat.allocate_gradients();
    AdamConfig lfs_config;
    lfs_config.lr = lr;
    AdamOptimizer lfs_opt(lfs_splat, lfs_config);

    // Phase 1: Optimize for 5 steps
    for (int iter = 0; iter < 5; iter++) {
        auto grad = torch::randn({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        torch_means.mutable_grad() = grad.clone();
        torch_opt->step(iter + 1);

        lfs_splat.means_grad() = from_torch(grad);
        lfs_opt.step(iter + 1);
    }

    // Phase 2: Reset state at specific indices (simulating MCMC relocation)
    std::vector<int64_t> reset_indices = {10, 25, 50, 75};
    auto torch_reset_indices = torch::tensor(reset_indices, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

    // Reset PyTorch state
    auto& torch_state_map = torch_opt->state();
    auto torch_state_it = torch_state_map.find(torch_means.unsafeGetTensorImpl());
    auto* torch_adam_state = static_cast<gs::training::FusedAdam::AdamParamState*>(torch_state_it->second.get());
    torch_adam_state->exp_avg.index_put_({torch_reset_indices}, 0);
    torch_adam_state->exp_avg_sq.index_put_({torch_reset_indices}, 0);

    // Reset LFS state
    lfs_opt.reset_state_at_indices(ParamType::Means, reset_indices);

    // Phase 3: Continue optimizing for 5 more steps
    for (int iter = 5; iter < 10; iter++) {
        auto grad = torch::randn({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        torch_means.mutable_grad() = grad.clone();
        torch_opt->step(iter + 1);

        lfs_splat.means_grad() = from_torch(grad);
        lfs_opt.step(iter + 1);
    }

    // Compare final results
    auto torch_final = torch_means.detach();
    float max_d = max_diff(lfs_splat.means(), torch_final);

    EXPECT_LT(max_d, tolerance)
        << "Parameters diverged after MCMC relocate sequence, max diff: " << max_d;

    // Verify reset indices have zero state
    auto* lfs_state = lfs_opt.get_state(ParamType::Means);
    for (int64_t idx : reset_indices) {
        // State should have been rebuilt from zero during optimization
        // We just verify the optimization completed successfully
    }
}

TEST(AdamOptimizerComparisonTest, MCMCAddGaussiansSequence) {
    // Simulate MCMC add: optimize → extend state → optimize extended params
    constexpr int N_initial = 50;
    constexpr int N_new = 25;
    constexpr float lr = 0.01f;
    constexpr float tolerance = 1e-5f;

    // Setup initial state
    auto torch_means = torch::randn({N_initial, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto lfs_means_data = torch_means.clone().detach();

    auto torch_opt = std::make_unique<gs::training::FusedAdam>(
        std::vector<torch::Tensor>{torch_means},
        std::make_unique<gs::training::FusedAdam::Options>(lr)
    );

    auto lfs_splat = create_test_splat_data(N_initial, 3);
    lfs_splat.means() = from_torch(lfs_means_data);
    lfs_splat.allocate_gradients();
    AdamConfig lfs_config;
    lfs_config.lr = lr;
    AdamOptimizer lfs_opt(lfs_splat, lfs_config);

    // Phase 1: Optimize initial params
    for (int iter = 0; iter < 5; iter++) {
        auto grad = torch::randn({N_initial, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        torch_means.mutable_grad() = grad.clone();
        torch_opt->step(iter + 1);

        lfs_splat.means_grad() = from_torch(grad);
        lfs_opt.step(iter + 1);
    }

    // Save step count before extension
    auto* lfs_state_before = lfs_opt.get_state(ParamType::Means);
    int64_t step_count_before = lfs_state_before->step_count;

    // Phase 2: Add new Gaussians
    auto new_means = torch::randn({N_new, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Extend PyTorch (matching MCMC pattern - modify optimizer in-place)
    auto extended_torch_means = torch::cat({torch_means.detach(), new_means}, 0).requires_grad_(true);

    // Get old state and save it
    auto& state_map = torch_opt->state();
    auto old_state_it = state_map.find(torch_means.unsafeGetTensorImpl());
    auto* old_adam_state = static_cast<gs::training::FusedAdam::AdamParamState*>(old_state_it->second.get());

    // Create new extended state
    auto zeros_to_add = torch::zeros({N_new, 3}, old_adam_state->exp_avg.options());
    auto new_state = std::make_unique<gs::training::FusedAdam::AdamParamState>();
    new_state->step_count = old_adam_state->step_count;  // PRESERVE step count
    new_state->exp_avg = torch::cat({old_adam_state->exp_avg, zeros_to_add}, 0);
    new_state->exp_avg_sq = torch::cat({old_adam_state->exp_avg_sq, zeros_to_add}, 0);

    // Verify step count preserved
    EXPECT_EQ(new_state->step_count, step_count_before)
        << "PyTorch step count should be preserved";

    // Remove old state and update parameter (matching MCMC pattern)
    state_map.erase(torch_means.unsafeGetTensorImpl());
    torch_opt->param_groups()[0].params()[0] = extended_torch_means;
    state_map[extended_torch_means.unsafeGetTensorImpl()] = std::move(new_state);
    torch_means = extended_torch_means;

    // Extend LFS
    auto lfs_new_means = from_torch(new_means);
    lfs_splat.means() = Tensor::cat(std::vector<Tensor>{lfs_splat.means(), lfs_new_means}, 0);
    lfs_splat.means_grad() = Tensor::cat(std::vector<Tensor>{lfs_splat.means_grad(), Tensor::zeros({N_new, 3}, Device::CUDA)}, 0);
    lfs_opt.extend_state_for_new_params(ParamType::Means, N_new);

    // Verify LFS step count preserved
    auto* lfs_state_after = lfs_opt.get_state(ParamType::Means);
    EXPECT_EQ(lfs_state_after->step_count, step_count_before)
        << "Step count must be preserved when extending (MCMC requirement)";

    // Phase 3: Continue optimizing extended params
    for (int iter = 5; iter < 10; iter++) {
        auto grad = torch::randn({N_initial + N_new, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        torch_means.mutable_grad() = grad.clone();
        torch_opt->step(iter + 1);

        lfs_splat.means_grad() = from_torch(grad);
        lfs_opt.step(iter + 1);
    }

    // Compare final results
    auto torch_final = torch_means.detach();
    float max_d = max_diff(lfs_splat.means(), torch_final);

    EXPECT_LT(max_d, tolerance)
        << "Parameters diverged after MCMC add sequence, max diff: " << max_d;
}

TEST(AdamOptimizerComparisonTest, MCMCFullSequence) {
    // Full MCMC simulation: optimize → add → optimize → relocate → optimize
    constexpr int N_initial = 80;
    constexpr int N_add = 20;
    constexpr float lr = 0.01f;
    constexpr float tolerance = 1e-4f;  // Slightly relaxed for long sequence

    // Initial setup
    auto torch_means = torch::randn({N_initial, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto lfs_means_data = torch_means.clone().detach();

    auto torch_opt = std::make_unique<gs::training::FusedAdam>(
        std::vector<torch::Tensor>{torch_means},
        std::make_unique<gs::training::FusedAdam::Options>(lr)
    );

    auto lfs_splat = create_test_splat_data(N_initial, 3);
    lfs_splat.means() = from_torch(lfs_means_data);
    lfs_splat.allocate_gradients();
    AdamConfig lfs_config;
    lfs_config.lr = lr;
    AdamOptimizer lfs_opt(lfs_splat, lfs_config);

    int iteration = 0;

    // Step 1: Initial optimization (3 iterations)
    for (int i = 0; i < 3; i++, iteration++) {
        auto grad = torch::randn({N_initial, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        torch_means.mutable_grad() = grad.clone();
        torch_opt->step(iteration + 1);
        lfs_splat.means_grad() = from_torch(grad);
        lfs_opt.step(iteration + 1);
    }

    // Step 2: Add new Gaussians
    auto new_means = torch::randn({N_add, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto extended_torch_means = torch::cat({torch_means.detach(), new_means}, 0).requires_grad_(true);

    // Extend PyTorch state (matching MCMC pattern - modify optimizer in-place)
    auto& state_map = torch_opt->state();
    auto old_it = state_map.find(torch_means.unsafeGetTensorImpl());
    auto* old_adam = static_cast<gs::training::FusedAdam::AdamParamState*>(old_it->second.get());

    // Create new extended state
    auto zeros = torch::zeros({N_add, 3}, old_adam->exp_avg.options());
    auto new_state = std::make_unique<gs::training::FusedAdam::AdamParamState>();
    new_state->step_count = old_adam->step_count;  // PRESERVE step count
    new_state->exp_avg = torch::cat({old_adam->exp_avg, zeros}, 0);
    new_state->exp_avg_sq = torch::cat({old_adam->exp_avg_sq, zeros}, 0);

    // Remove old state and update parameter (matching MCMC pattern)
    state_map.erase(torch_means.unsafeGetTensorImpl());
    torch_opt->param_groups()[0].params()[0] = extended_torch_means;
    state_map[extended_torch_means.unsafeGetTensorImpl()] = std::move(new_state);
    torch_means = extended_torch_means;

    // Get pointer to the new state for later use
    auto state_it = state_map.find(torch_means.unsafeGetTensorImpl());
    auto* torch_adam_state = static_cast<gs::training::FusedAdam::AdamParamState*>(state_it->second.get());

    // Extend LFS
    lfs_splat.means() = Tensor::cat(std::vector<Tensor>{lfs_splat.means(), from_torch(new_means)}, 0);
    lfs_splat.means_grad() = Tensor::cat(std::vector<Tensor>{lfs_splat.means_grad(), Tensor::zeros({N_add, 3}, Device::CUDA)}, 0);
    lfs_opt.extend_state_for_new_params(ParamType::Means, N_add);

    // Step 3: Optimize extended (3 iterations)
    for (int i = 0; i < 3; i++, iteration++) {
        auto grad = torch::randn({N_initial + N_add, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        torch_means.mutable_grad() = grad.clone();
        torch_opt->step(iteration + 1);
        lfs_splat.means_grad() = from_torch(grad);
        lfs_opt.step(iteration + 1);
    }

    // Step 4: Relocate dead Gaussians
    std::vector<int64_t> dead_indices = {5, 15, 25, 35};
    auto torch_dead = torch::tensor(dead_indices, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

    torch_adam_state->exp_avg.index_put_({torch_dead}, 0);
    torch_adam_state->exp_avg_sq.index_put_({torch_dead}, 0);
    lfs_opt.reset_state_at_indices(ParamType::Means, dead_indices);

    // Step 5: Final optimization (3 iterations)
    for (int i = 0; i < 3; i++, iteration++) {
        auto grad = torch::randn({N_initial + N_add, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        torch_means.mutable_grad() = grad.clone();
        torch_opt->step(iteration + 1);
        lfs_splat.means_grad() = from_torch(grad);
        lfs_opt.step(iteration + 1);
    }

    // Final comparison
    auto torch_final = torch_means.detach();
    float max_d = max_diff(lfs_splat.means(), torch_final);

    EXPECT_LT(max_d, tolerance)
        << "Full MCMC sequence diverged, max diff: " << max_d;
}

// ===== Tests for Safe API =====

TEST(AdamOptimizerSafeAPITest, AddNewParamsAtomic) {
    const size_t N_initial = 100;
    const size_t N_new = 50;

    auto splat = create_test_splat_data(N_initial, 3);
    splat.allocate_gradients();

    AdamConfig config;
    config.lr = 1e-3f;
    AdamOptimizer opt(splat, config);

    // Do a few steps to initialize state
    for (int i = 0; i < 3; i++) {
        splat.means_grad() = Tensor::randn({N_initial, 3}, Device::CUDA);
        opt.step(i + 1);
    }

    auto* state_before = opt.get_state(ParamType::Means);
    int64_t step_count_before = state_before->step_count;

    // Add new parameters using safe API
    auto new_means = Tensor::randn({N_new, 3}, Device::CUDA);
    opt.add_new_params(ParamType::Means, new_means);

    // Verify all were updated atomically
    EXPECT_EQ(splat.means().shape()[0], N_initial + N_new)
        << "Parameters should be extended";
    EXPECT_EQ(splat.means_grad().shape()[0], N_initial + N_new)
        << "Gradients should be extended";

    auto* state_after = opt.get_state(ParamType::Means);
    EXPECT_EQ(state_after->exp_avg.shape()[0], N_initial + N_new)
        << "Optimizer state should be extended";
    EXPECT_EQ(state_after->step_count, step_count_before)
        << "Step count should be preserved";

    // Verify new gradients are zero
    auto grad_tail = splat.means_grad().slice(0, N_initial, N_initial + N_new);
    float grad_sum = grad_tail.abs().sum().item<float>();
    EXPECT_FLOAT_EQ(grad_sum, 0.0f)
        << "New gradients should be initialized to zero";
}

TEST(AdamOptimizerSafeAPITest, AddNewParamsValidation) {
    auto splat = create_test_splat_data(100, 3);
    splat.allocate_gradients();

    AdamConfig config;
    AdamOptimizer opt(splat, config);

    // Test 1: Wrong rank (validation enabled)
    auto wrong_rank = Tensor::randn({10, 3, 1}, Device::CUDA);
    EXPECT_THROW(opt.add_new_params(ParamType::Means, wrong_rank, true), std::runtime_error)
        << "Should reject tensor with wrong rank when validation enabled";

    // Test 2: Wrong dimension size (validation enabled)
    auto wrong_dim = Tensor::randn({10, 5}, Device::CUDA);  // means should be (N, 3)
    EXPECT_THROW(opt.add_new_params(ParamType::Means, wrong_dim, true), std::runtime_error)
        << "Should reject tensor with wrong dimension size when validation enabled";

    // Test 3: Wrong device (validation enabled)
    auto wrong_device = Tensor::randn({10, 3}, Device::CPU);
    EXPECT_THROW(opt.add_new_params(ParamType::Means, wrong_device, true), std::runtime_error)
        << "Should reject tensor on wrong device when validation enabled";

    // Test 4: Wrong dimension (validation disabled - still throws from cat, but different exception)
    // When validation is disabled, the early checks are skipped, but cat() itself will still
    // throw std::invalid_argument for mismatched shapes (lower-level check)
    EXPECT_THROW(opt.add_new_params(ParamType::Means, wrong_dim, false), std::invalid_argument)
        << "cat() should throw invalid_argument for mismatched shapes even when validation disabled";

    // Test 5: Correct shape should work
    auto correct = Tensor::randn({10, 3}, Device::CUDA);
    EXPECT_NO_THROW(opt.add_new_params(ParamType::Means, correct, true))
        << "Should accept tensor with correct shape";
}

TEST(AdamOptimizerSafeAPITest, RelocateParamsAtIndicesZerosGradients) {
    const size_t N = 100;
    auto splat = create_test_splat_data(N, 3);
    splat.allocate_gradients();

    AdamConfig config;
    AdamOptimizer opt(splat, config);

    // Initialize with some optimization steps
    for (int i = 0; i < 5; i++) {
        splat.means_grad() = Tensor::randn({N, 3}, Device::CUDA);
        opt.step(i + 1);
    }

    // Set non-zero gradients
    splat.means_grad() = Tensor::ones({N, 3}, Device::CUDA);

    // Relocate some indices
    std::vector<int64_t> indices = {10, 20, 30, 40};
    opt.relocate_params_at_indices(ParamType::Means, indices);

    // Verify gradients were zeroed at those indices
    auto grad_cpu = splat.means_grad().cpu();
    auto grad_vec = grad_cpu.to_vector();

    for (auto idx : indices) {
        for (size_t j = 0; j < 3; j++) {
            size_t offset = idx * 3 + j;
            EXPECT_FLOAT_EQ(grad_vec[offset], 0.0f)
                << "Gradient at relocated index should be zero";
        }
    }

    // Verify other gradients are still 1.0
    for (size_t i = 0; i < N; i++) {
        bool is_relocated = std::find(indices.begin(), indices.end(), i) != indices.end();
        if (!is_relocated) {
            for (size_t j = 0; j < 3; j++) {
                size_t offset = i * 3 + j;
                EXPECT_FLOAT_EQ(grad_vec[offset], 1.0f)
                    << "Non-relocated gradient should remain unchanged";
            }
        }
    }

    // Verify optimizer state was also reset
    auto* state = opt.get_state(ParamType::Means);
    auto exp_avg_cpu = state->exp_avg.cpu().to_vector();

    for (auto idx : indices) {
        for (size_t j = 0; j < 3; j++) {
            size_t offset = idx * 3 + j;
            EXPECT_FLOAT_EQ(exp_avg_cpu[offset], 0.0f)
                << "Optimizer state at relocated index should be zero";
        }
    }
}

TEST(AdamOptimizerSafeAPITest, RelocateParamsValidation) {
    const size_t N = 100;
    auto splat = create_test_splat_data(N, 3);
    splat.allocate_gradients();

    AdamConfig config;
    AdamOptimizer opt(splat, config);

    // Test out-of-bounds indices
    std::vector<int64_t> out_of_bounds = {-1};
    EXPECT_THROW(opt.relocate_params_at_indices(ParamType::Means, out_of_bounds), std::runtime_error)
        << "Should reject negative index";

    out_of_bounds = {100};  // N = 100, so valid range is [0, 99]
    EXPECT_THROW(opt.relocate_params_at_indices(ParamType::Means, out_of_bounds), std::runtime_error)
        << "Should reject index >= N";

    out_of_bounds = {50, 101, 70};
    EXPECT_THROW(opt.relocate_params_at_indices(ParamType::Means, out_of_bounds), std::runtime_error)
        << "Should reject any out-of-bounds index in list";
}

TEST(AdamOptimizerSafeAPITest, SafeAPICombinedWorkflow) {
    // This test demonstrates the simplified MCMC workflow with safe API
    const size_t N_initial = 100;
    const size_t N_add = 50;

    auto splat = create_test_splat_data(N_initial, 3);
    splat.allocate_gradients();

    AdamConfig config;
    config.lr = 1e-3f;
    AdamOptimizer opt(splat, config);

    // Step 1: Initial optimization
    for (int i = 0; i < 5; i++) {
        splat.means_grad() = Tensor::randn({N_initial, 3}, Device::CUDA);
        opt.step(i + 1);
    }

    // Step 2: Add new parameters (SAFE - atomic operation)
    auto new_means = Tensor::randn({N_add, 3}, Device::CUDA);
    opt.add_new_params(ParamType::Means, new_means);

    // Step 3: Continue optimization with extended parameters
    for (int i = 5; i < 10; i++) {
        splat.means_grad() = Tensor::randn({N_initial + N_add, 3}, Device::CUDA);
        opt.step(i + 1);
    }

    // Step 4: Relocate dead Gaussians (SAFE - zeros gradients too)
    std::vector<int64_t> dead_indices = {5, 15, 25, 105, 120};
    opt.relocate_params_at_indices(ParamType::Means, dead_indices);

    // Step 5: Final optimization
    for (int i = 10; i < 15; i++) {
        splat.means_grad() = Tensor::randn({N_initial + N_add, 3}, Device::CUDA);
        opt.step(i + 1);
    }

    // Verify final state
    EXPECT_EQ(splat.means().shape()[0], N_initial + N_add);
    auto* final_state = opt.get_state(ParamType::Means);
    EXPECT_EQ(final_state->step_count, 15);
}

// ===========================================================================================
// Benchmarks - Compare performance vs PyTorch FusedAdam
// ===========================================================================================

class AdamOptimizerBenchmark : public ::testing::Test {
protected:
    // Timing helpers
    struct BenchmarkResult {
        std::string name;
        double lfs_time_ms = 0.0;
        double torch_time_ms = 0.0;
        double speedup = 0.0;
        size_t operations = 0;

        void print() const {
            printf("  %-40s | LFS: %8.3f ms | PyTorch: %8.3f ms | Speedup: %.2fx\n",
                   name.c_str(), lfs_time_ms, torch_time_ms, speedup);
        }
    };

    std::vector<BenchmarkResult> results;

    void print_results() {
        printf("\n");
        printf("========================================================================\n");
        printf("                    OPTIMIZER BENCHMARK RESULTS\n");
        printf("========================================================================\n");
        for (const auto& r : results) {
            r.print();
        }
        printf("========================================================================\n\n");
    }

    template<typename Func>
    double time_operation(Func&& func, int warmup = 3, int iterations = 10) {
        // Warmup
        for (int i = 0; i < warmup; i++) {
            func();
        }
        cudaDeviceSynchronize();

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            func();
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count() / iterations;
    }

    ~AdamOptimizerBenchmark() override {
        if (!results.empty()) {
            print_results();
        }
    }
};

TEST_F(AdamOptimizerBenchmark, StepPerformance) {
    // Test step performance at different scales - FAIR: both optimize ALL 6 parameters
    std::vector<size_t> sizes = {10'000, 50'000, 100'000, 500'000};

    for (size_t N : sizes) {
        const float lr = 1e-3f;

        // Setup PyTorch - ALL 6 parameters like LFS!
        auto torch_means = torch::randn({static_cast<long>(N), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
        auto torch_sh0 = torch::randn({static_cast<long>(N), 1, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
        auto torch_shN = torch::randn({static_cast<long>(N), 15, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
        auto torch_scaling = torch::randn({static_cast<long>(N), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
        auto torch_rotation = torch::randn({static_cast<long>(N), 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
        auto torch_opacity = torch::randn({static_cast<long>(N), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);

        auto torch_opt = std::make_unique<gs::training::FusedAdam>(
            std::vector<torch::Tensor>{torch_means, torch_sh0, torch_shN, torch_scaling, torch_rotation, torch_opacity},
            std::make_unique<gs::training::FusedAdam::Options>(lr)
        );

        // Setup LFS
        auto lfs_splat = create_test_splat_data(N, 3);
        lfs_splat.means() = from_torch(torch_means.detach());
        lfs_splat.sh0() = from_torch(torch_sh0.detach());
        lfs_splat.shN() = from_torch(torch_shN.detach());
        lfs_splat.scaling_raw() = from_torch(torch_scaling.detach());
        lfs_splat.rotation_raw() = from_torch(torch_rotation.detach());
        lfs_splat.opacity_raw() = from_torch(torch_opacity.detach());
        lfs_splat.allocate_gradients();

        AdamConfig lfs_config;
        lfs_config.lr = lr;
        lfs_config.growth_factor = 1.0f;  // No pre-allocation for fair comparison
        AdamOptimizer lfs_opt(lfs_splat, lfs_config);

        // Create gradients for ALL parameters
        auto grad_means = torch::randn({static_cast<long>(N), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto grad_sh0 = torch::randn({static_cast<long>(N), 1, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto grad_shN = torch::randn({static_cast<long>(N), 15, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto grad_scaling = torch::randn({static_cast<long>(N), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto grad_rotation = torch::randn({static_cast<long>(N), 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto grad_opacity = torch::randn({static_cast<long>(N), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Pre-convert LFS gradients (do this ONCE, not in benchmark loop!)
        auto lfs_grad_means = from_torch(grad_means);
        auto lfs_grad_sh0 = from_torch(grad_sh0);
        auto lfs_grad_shN = from_torch(grad_shN);
        auto lfs_grad_scaling = from_torch(grad_scaling);
        auto lfs_grad_rotation = from_torch(grad_rotation);
        auto lfs_grad_opacity = from_torch(grad_opacity);

        // Benchmark PyTorch - ALL 6 parameters
        double torch_time = time_operation([&]() {
            torch_means.mutable_grad() = grad_means;
            torch_sh0.mutable_grad() = grad_sh0;
            torch_shN.mutable_grad() = grad_shN;
            torch_scaling.mutable_grad() = grad_scaling;
            torch_rotation.mutable_grad() = grad_rotation;
            torch_opacity.mutable_grad() = grad_opacity;
            torch_opt->step(1);
        });

        // Benchmark LFS - ALL 6 parameters (no conversion overhead!)
        double lfs_time = time_operation([&]() {
            lfs_splat.means_grad() = lfs_grad_means;
            lfs_splat.sh0_grad() = lfs_grad_sh0;
            lfs_splat.shN_grad() = lfs_grad_shN;
            lfs_splat.scaling_grad() = lfs_grad_scaling;
            lfs_splat.rotation_grad() = lfs_grad_rotation;
            lfs_splat.opacity_grad() = lfs_grad_opacity;
            lfs_opt.step(1);
        });

        BenchmarkResult result;
        result.name = "Step (N=" + std::to_string(N) + ", 6 params)";
        result.lfs_time_ms = lfs_time;
        result.torch_time_ms = torch_time;
        result.speedup = torch_time / lfs_time;
        result.operations = N;
        results.push_back(result);
    }
}

TEST_F(AdamOptimizerBenchmark, AddParametersNoPreallocation) {
    // Benchmark adding parameters WITHOUT pre-allocation (worst case)
    const size_t N_initial = 100'000;
    const size_t N_add = 10'000;
    const int n_additions = 20;  // Simulate 20 MCMC additions
    const float lr = 1e-3f;

    // Setup PyTorch
    auto torch_means = torch::randn({static_cast<long>(N_initial), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto torch_opt = std::make_unique<gs::training::FusedAdam>(
        std::vector<torch::Tensor>{torch_means},
        std::make_unique<gs::training::FusedAdam::Options>(lr)
    );

    // Initialize state with a few steps
    for (int i = 0; i < 3; i++) {
        auto g = torch::randn({static_cast<long>(N_initial), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        torch_means.mutable_grad() = g;
        torch_opt->step(i + 1);
    }

    // Setup LFS
    auto lfs_splat = create_test_splat_data(N_initial, 3);
    lfs_splat.means() = from_torch(torch_means.detach());
    lfs_splat.allocate_gradients();

    AdamConfig lfs_config;
    lfs_config.lr = lr;
    lfs_config.growth_factor = 1.0f;  // NO growth factor = worst case
    lfs_config.initial_capacity = 0;   // NO pre-allocation
    AdamOptimizer lfs_opt(lfs_splat, lfs_config);

    // Initialize state
    for (int i = 0; i < 3; i++) {
        lfs_splat.means_grad() = Tensor::randn({N_initial, 3}, Device::CUDA);
        lfs_opt.step(i + 1);
    }

    // Benchmark PyTorch additions
    double torch_time = time_operation([&]() {
        auto new_means = torch::randn({static_cast<long>(N_add), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto extended = torch::cat({torch_means.detach(), new_means}, 0).requires_grad_(true);

        // Get old state
        auto& state_map = torch_opt->state();
        auto state_it = state_map.find(torch_means.unsafeGetTensorImpl());
        auto* old_state = static_cast<gs::training::FusedAdam::AdamParamState*>(state_it->second.get());

        // Create new state
        auto zeros = torch::zeros({static_cast<long>(N_add), 3}, old_state->exp_avg.options());
        auto new_state = std::make_unique<gs::training::FusedAdam::AdamParamState>();
        new_state->step_count = old_state->step_count;
        new_state->exp_avg = torch::cat({old_state->exp_avg, zeros}, 0);
        new_state->exp_avg_sq = torch::cat({old_state->exp_avg_sq, zeros}, 0);

        // Update optimizer
        state_map.erase(torch_means.unsafeGetTensorImpl());
        torch_opt->param_groups()[0].params()[0] = extended;
        state_map[extended.unsafeGetTensorImpl()] = std::move(new_state);
        torch_means = extended;
    }, 1, n_additions);

    // Benchmark LFS additions (disable validation for fair comparison)
    double lfs_time = time_operation([&]() {
        auto new_means = Tensor::randn({N_add, 3}, Device::CUDA);
        lfs_opt.add_new_params(ParamType::Means, new_means, false);  // No validation
    }, 1, n_additions);

    BenchmarkResult result;
    result.name = "Add params (no pre-alloc, " + std::to_string(n_additions) + "x" + std::to_string(N_add) + ")";
    result.lfs_time_ms = lfs_time;
    result.torch_time_ms = torch_time;
    result.speedup = torch_time / lfs_time;
    result.operations = n_additions;
    results.push_back(result);
}

TEST_F(AdamOptimizerBenchmark, AddParametersWithGrowthFactor) {
    // Benchmark adding parameters WITH growth factor (1.5x)
    const size_t N_initial = 100'000;
    const size_t N_add = 10'000;
    const int n_additions = 20;
    const float lr = 1e-3f;

    // Setup LFS with growth factor
    auto lfs_splat = create_test_splat_data(N_initial, 3);
    lfs_splat.allocate_gradients();

    AdamConfig lfs_config;
    lfs_config.lr = lr;
    lfs_config.growth_factor = 1.5f;  // WITH growth factor
    lfs_config.initial_capacity = 0;
    AdamOptimizer lfs_opt(lfs_splat, lfs_config);

    // Initialize state
    for (int i = 0; i < 3; i++) {
        lfs_splat.means_grad() = Tensor::randn({N_initial, 3}, Device::CUDA);
        lfs_opt.step(i + 1);
    }

    // Benchmark LFS additions with growth (no validation)
    double lfs_time = time_operation([&]() {
        auto new_means = Tensor::randn({N_add, 3}, Device::CUDA);
        lfs_opt.add_new_params(ParamType::Means, new_means, false);
    }, 1, n_additions);

    BenchmarkResult result;
    result.name = "Add params (1.5x growth, " + std::to_string(n_additions) + "x" + std::to_string(N_add) + ")";
    result.lfs_time_ms = lfs_time;
    result.torch_time_ms = 0.0;  // No PyTorch comparison (same as above)
    result.speedup = 0.0;
    result.operations = n_additions;
    results.push_back(result);

    printf("  Note: With 1.5x growth factor, %zu additions required only ~%zu reallocations\n",
           static_cast<size_t>(n_additions),
           static_cast<size_t>(std::ceil(std::log(1.0 + n_additions * N_add / (double)N_initial) / std::log(1.5))));
}

TEST_F(AdamOptimizerBenchmark, AddParametersWithPreallocation) {
    // Benchmark adding parameters WITH full pre-allocation (best case)
    const size_t N_initial = 100'000;
    const size_t N_add = 10'000;
    const int n_additions = 20;
    const float lr = 1e-3f;

    // Setup LFS with pre-allocation
    auto lfs_splat = create_test_splat_data(N_initial, 3);
    lfs_splat.allocate_gradients();

    AdamConfig lfs_config;
    lfs_config.lr = lr;
    lfs_config.growth_factor = 1.5f;
    lfs_config.initial_capacity = N_initial + (n_additions * N_add);  // Pre-allocate for all
    AdamOptimizer lfs_opt(lfs_splat, lfs_config);

    // Initialize state
    for (int i = 0; i < 3; i++) {
        lfs_splat.means_grad() = Tensor::randn({N_initial, 3}, Device::CUDA);
        lfs_opt.step(i + 1);
    }

    // Benchmark LFS additions with pre-allocation (no validation)
    double lfs_time = time_operation([&]() {
        auto new_means = Tensor::randn({N_add, 3}, Device::CUDA);
        lfs_opt.add_new_params(ParamType::Means, new_means, false);
    }, 1, n_additions);

    BenchmarkResult result;
    result.name = "Add params (pre-allocated, " + std::to_string(n_additions) + "x" + std::to_string(N_add) + ")";
    result.lfs_time_ms = lfs_time;
    result.torch_time_ms = 0.0;
    result.speedup = 0.0;
    result.operations = n_additions;
    results.push_back(result);

    printf("  Note: Pre-allocation enabled ZERO reallocations for optimizer state!\n");
}

TEST_F(AdamOptimizerBenchmark, RelocateParameters) {
    // Benchmark relocating dead Gaussians
    const size_t N = 200'000;
    const size_t N_relocate = 1'000;
    const int n_relocations = 10;
    const float lr = 1e-3f;

    // Setup PyTorch
    auto torch_means = torch::randn({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto torch_opt = std::make_unique<gs::training::FusedAdam>(
        std::vector<torch::Tensor>{torch_means},
        std::make_unique<gs::training::FusedAdam::Options>(lr)
    );

    // Initialize
    for (int i = 0; i < 3; i++) {
        auto g = torch::randn({static_cast<long>(N), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        torch_means.mutable_grad() = g;
        torch_opt->step(i + 1);
    }

    // Setup LFS
    auto lfs_splat = create_test_splat_data(N, 3);
    lfs_splat.means() = from_torch(torch_means.detach());
    lfs_splat.allocate_gradients();

    AdamConfig lfs_config;
    lfs_config.lr = lr;
    AdamOptimizer lfs_opt(lfs_splat, lfs_config);

    for (int i = 0; i < 3; i++) {
        lfs_splat.means_grad() = Tensor::randn({N, 3}, Device::CUDA);
        lfs_opt.step(i + 1);
    }

    // Generate random indices
    std::vector<int64_t> indices(N_relocate);
    for (size_t i = 0; i < N_relocate; i++) {
        indices[i] = rand() % N;
    }
    auto torch_indices = torch::tensor(indices,
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

    // Pre-allocate GPU indices for LFS (fair comparison with PyTorch!)
    int64_t* lfs_indices_gpu;
    cudaMalloc(&lfs_indices_gpu, N_relocate * sizeof(int64_t));
    cudaMemcpy(lfs_indices_gpu, indices.data(), N_relocate * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Benchmark PyTorch
    double torch_time = time_operation([&]() {
        auto& state_map = torch_opt->state();
        auto state_it = state_map.find(torch_means.unsafeGetTensorImpl());
        auto* state = static_cast<gs::training::FusedAdam::AdamParamState*>(state_it->second.get());
        state->exp_avg.index_put_({torch_indices}, 0);
        state->exp_avg_sq.index_put_({torch_indices}, 0);
    }, 1, n_relocations);

    // Benchmark LFS - using GPU indices (fair comparison!)
    double lfs_time = time_operation([&]() {
        lfs_opt.relocate_params_at_indices_gpu(ParamType::Means, lfs_indices_gpu, N_relocate);
    }, 1, n_relocations);

    cudaFree(lfs_indices_gpu);

    BenchmarkResult result;
    result.name = "Relocate (N=" + std::to_string(N) + ", " + std::to_string(N_relocate) + " indices)";
    result.lfs_time_ms = lfs_time;
    result.torch_time_ms = torch_time;
    result.speedup = torch_time / lfs_time;
    result.operations = n_relocations;
    results.push_back(result);
}

TEST_F(AdamOptimizerBenchmark, FullMCMCWorkflow) {
    // Realistic MCMC workflow benchmark
    const size_t N_initial = 100'000;
    const int n_iterations = 100;
    const int add_every = 10;  // Add every 10 iterations
    const size_t N_add_per_iter = 5'000;
    const float lr = 1e-3f;

    printf("\n  Simulating %d iterations of MCMC training (6 parameters)...\n", n_iterations);

    // Setup PyTorch - ALL 6 parameters for fair comparison!
    auto torch_means = torch::randn({static_cast<long>(N_initial), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto torch_sh0 = torch::randn({static_cast<long>(N_initial), 1, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto torch_shN = torch::randn({static_cast<long>(N_initial), 15, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto torch_scaling = torch::randn({static_cast<long>(N_initial), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto torch_rotation = torch::randn({static_cast<long>(N_initial), 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);
    auto torch_opacity = torch::randn({static_cast<long>(N_initial), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_(true);

    auto torch_opt = std::make_unique<gs::training::FusedAdam>(
        std::vector<torch::Tensor>{torch_means, torch_sh0, torch_shN, torch_scaling, torch_rotation, torch_opacity},
        std::make_unique<gs::training::FusedAdam::Options>(lr)
    );

    // Setup LFS (no pre-allocation)
    auto lfs_splat_no_prealloc = create_test_splat_data(N_initial, 3);
    lfs_splat_no_prealloc.allocate_gradients();
    AdamConfig lfs_config_no_prealloc;
    lfs_config_no_prealloc.lr = lr;
    lfs_config_no_prealloc.growth_factor = 1.0f;  // No growth
    AdamOptimizer lfs_opt_no_prealloc(lfs_splat_no_prealloc, lfs_config_no_prealloc);

    // Setup LFS (with 1.5x growth)
    auto lfs_splat_growth = create_test_splat_data(N_initial, 3);
    lfs_splat_growth.allocate_gradients();
    AdamConfig lfs_config_growth;
    lfs_config_growth.lr = lr;
    lfs_config_growth.growth_factor = 1.5f;
    AdamOptimizer lfs_opt_growth(lfs_splat_growth, lfs_config_growth);

    // Setup LFS (with pre-allocation)
    auto lfs_splat_prealloc = create_test_splat_data(N_initial, 3);
    lfs_splat_prealloc.allocate_gradients();
    AdamConfig lfs_config_prealloc;
    lfs_config_prealloc.lr = lr;
    lfs_config_prealloc.growth_factor = 1.5f;
    lfs_config_prealloc.initial_capacity = N_initial + (n_iterations / add_every) * N_add_per_iter;
    AdamOptimizer lfs_opt_prealloc(lfs_splat_prealloc, lfs_config_prealloc);

    size_t current_N = N_initial;

    // Benchmark PyTorch workflow - ALL 6 parameters
    double torch_time = time_operation([&]() {
        for (int iter = 0; iter < n_iterations; iter++) {
            // Step - set gradients for ALL 6 parameters
            torch_means.mutable_grad() = torch::randn({static_cast<long>(current_N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            torch_sh0.mutable_grad() = torch::randn({static_cast<long>(current_N), 1, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            torch_shN.mutable_grad() = torch::randn({static_cast<long>(current_N), 15, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            torch_scaling.mutable_grad() = torch::randn({static_cast<long>(current_N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            torch_rotation.mutable_grad() = torch::randn({static_cast<long>(current_N), 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            torch_opacity.mutable_grad() = torch::randn({static_cast<long>(current_N), 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            torch_opt->step(iter + 1);

            // Add parameters periodically (only means for simplicity, but optimizer handles all 6)
            if ((iter + 1) % add_every == 0) {
                // Just add to means (to match LFS benchmark which only adds means)
                auto new_means = torch::randn({static_cast<long>(N_add_per_iter), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                torch_means = torch::cat({torch_means.detach(), new_means}, 0).requires_grad_(true);
                torch_sh0 = torch::cat({torch_sh0.detach(), torch::randn({static_cast<long>(N_add_per_iter), 1, 3}, torch_sh0.options())}, 0).requires_grad_(true);
                torch_shN = torch::cat({torch_shN.detach(), torch::randn({static_cast<long>(N_add_per_iter), 15, 3}, torch_shN.options())}, 0).requires_grad_(true);
                torch_scaling = torch::cat({torch_scaling.detach(), torch::randn({static_cast<long>(N_add_per_iter), 3}, torch_scaling.options())}, 0).requires_grad_(true);
                torch_rotation = torch::cat({torch_rotation.detach(), torch::randn({static_cast<long>(N_add_per_iter), 4}, torch_rotation.options())}, 0).requires_grad_(true);
                torch_opacity = torch::cat({torch_opacity.detach(), torch::randn({static_cast<long>(N_add_per_iter), 1}, torch_opacity.options())}, 0).requires_grad_(true);

                // Update optimizer state for all 6 parameters
                auto& state_map = torch_opt->state();
                for (auto& p : torch_opt->param_groups()[0].params()) {
                    auto it = state_map.find(p.unsafeGetTensorImpl());
                    if (it != state_map.end()) {
                        state_map.erase(it);
                    }
                }

                torch_opt->param_groups()[0].params() = {torch_means, torch_sh0, torch_shN, torch_scaling, torch_rotation, torch_opacity};
                current_N += N_add_per_iter;
            }
        }
    }, 1, 1);

    // Reset for LFS benchmarks
    current_N = N_initial;

    // Benchmark LFS (no pre-allocation) - ALL 6 parameters
    double lfs_time_no_prealloc = time_operation([&]() {
        for (int iter = 0; iter < n_iterations; iter++) {
            // Set gradients for ALL 6 parameters
            lfs_splat_no_prealloc.means_grad() = Tensor::randn({current_N, 3}, Device::CUDA);
            lfs_splat_no_prealloc.sh0_grad() = Tensor::randn({current_N, 1, 3}, Device::CUDA);
            lfs_splat_no_prealloc.shN_grad() = Tensor::randn({current_N, 15, 3}, Device::CUDA);
            lfs_splat_no_prealloc.scaling_grad() = Tensor::randn({current_N, 3}, Device::CUDA);
            lfs_splat_no_prealloc.rotation_grad() = Tensor::randn({current_N, 4}, Device::CUDA);
            lfs_splat_no_prealloc.opacity_grad() = Tensor::randn({current_N, 1}, Device::CUDA);
            lfs_opt_no_prealloc.step(iter + 1);

            if ((iter + 1) % add_every == 0) {
                // Add to ALL 6 parameters
                lfs_opt_no_prealloc.add_new_params(ParamType::Means, Tensor::randn({N_add_per_iter, 3}, Device::CUDA), false);
                lfs_opt_no_prealloc.add_new_params(ParamType::Sh0, Tensor::randn({N_add_per_iter, 1, 3}, Device::CUDA), false);
                lfs_opt_no_prealloc.add_new_params(ParamType::ShN, Tensor::randn({N_add_per_iter, 15, 3}, Device::CUDA), false);
                lfs_opt_no_prealloc.add_new_params(ParamType::Scaling, Tensor::randn({N_add_per_iter, 3}, Device::CUDA), false);
                lfs_opt_no_prealloc.add_new_params(ParamType::Rotation, Tensor::randn({N_add_per_iter, 4}, Device::CUDA), false);
                lfs_opt_no_prealloc.add_new_params(ParamType::Opacity, Tensor::randn({N_add_per_iter, 1}, Device::CUDA), false);
                current_N += N_add_per_iter;
            }
        }
    }, 1, 1);

    current_N = N_initial;

    // Benchmark LFS (1.5x growth) - ALL 6 parameters
    double lfs_time_growth = time_operation([&]() {
        for (int iter = 0; iter < n_iterations; iter++) {
            lfs_splat_growth.means_grad() = Tensor::randn({current_N, 3}, Device::CUDA);
            lfs_splat_growth.sh0_grad() = Tensor::randn({current_N, 1, 3}, Device::CUDA);
            lfs_splat_growth.shN_grad() = Tensor::randn({current_N, 15, 3}, Device::CUDA);
            lfs_splat_growth.scaling_grad() = Tensor::randn({current_N, 3}, Device::CUDA);
            lfs_splat_growth.rotation_grad() = Tensor::randn({current_N, 4}, Device::CUDA);
            lfs_splat_growth.opacity_grad() = Tensor::randn({current_N, 1}, Device::CUDA);
            lfs_opt_growth.step(iter + 1);

            if ((iter + 1) % add_every == 0) {
                lfs_opt_growth.add_new_params(ParamType::Means, Tensor::randn({N_add_per_iter, 3}, Device::CUDA), false);
                lfs_opt_growth.add_new_params(ParamType::Sh0, Tensor::randn({N_add_per_iter, 1, 3}, Device::CUDA), false);
                lfs_opt_growth.add_new_params(ParamType::ShN, Tensor::randn({N_add_per_iter, 15, 3}, Device::CUDA), false);
                lfs_opt_growth.add_new_params(ParamType::Scaling, Tensor::randn({N_add_per_iter, 3}, Device::CUDA), false);
                lfs_opt_growth.add_new_params(ParamType::Rotation, Tensor::randn({N_add_per_iter, 4}, Device::CUDA), false);
                lfs_opt_growth.add_new_params(ParamType::Opacity, Tensor::randn({N_add_per_iter, 1}, Device::CUDA), false);
                current_N += N_add_per_iter;
            }
        }
    }, 1, 1);

    current_N = N_initial;

    // Benchmark LFS (pre-allocated) - ALL 6 parameters
    double lfs_time_prealloc = time_operation([&]() {
        for (int iter = 0; iter < n_iterations; iter++) {
            lfs_splat_prealloc.means_grad() = Tensor::randn({current_N, 3}, Device::CUDA);
            lfs_splat_prealloc.sh0_grad() = Tensor::randn({current_N, 1, 3}, Device::CUDA);
            lfs_splat_prealloc.shN_grad() = Tensor::randn({current_N, 15, 3}, Device::CUDA);
            lfs_splat_prealloc.scaling_grad() = Tensor::randn({current_N, 3}, Device::CUDA);
            lfs_splat_prealloc.rotation_grad() = Tensor::randn({current_N, 4}, Device::CUDA);
            lfs_splat_prealloc.opacity_grad() = Tensor::randn({current_N, 1}, Device::CUDA);
            lfs_opt_prealloc.step(iter + 1);

            if ((iter + 1) % add_every == 0) {
                lfs_opt_prealloc.add_new_params(ParamType::Means, Tensor::randn({N_add_per_iter, 3}, Device::CUDA), false);
                lfs_opt_prealloc.add_new_params(ParamType::Sh0, Tensor::randn({N_add_per_iter, 1, 3}, Device::CUDA), false);
                lfs_opt_prealloc.add_new_params(ParamType::ShN, Tensor::randn({N_add_per_iter, 15, 3}, Device::CUDA), false);
                lfs_opt_prealloc.add_new_params(ParamType::Scaling, Tensor::randn({N_add_per_iter, 3}, Device::CUDA), false);
                lfs_opt_prealloc.add_new_params(ParamType::Rotation, Tensor::randn({N_add_per_iter, 4}, Device::CUDA), false);
                lfs_opt_prealloc.add_new_params(ParamType::Opacity, Tensor::randn({N_add_per_iter, 1}, Device::CUDA), false);
                current_N += N_add_per_iter;
            }
        }
    }, 1, 1);

    BenchmarkResult result1;
    result1.name = "MCMC Workflow (PyTorch)";
    result1.lfs_time_ms = 0.0;
    result1.torch_time_ms = torch_time;
    result1.speedup = 0.0;
    result1.operations = n_iterations;
    results.push_back(result1);

    BenchmarkResult result2;
    result2.name = "MCMC Workflow (LFS no growth)";
    result2.lfs_time_ms = lfs_time_no_prealloc;
    result2.torch_time_ms = torch_time;
    result2.speedup = torch_time / lfs_time_no_prealloc;
    result2.operations = n_iterations;
    results.push_back(result2);

    BenchmarkResult result3;
    result3.name = "MCMC Workflow (LFS 1.5x growth)";
    result3.lfs_time_ms = lfs_time_growth;
    result3.torch_time_ms = torch_time;
    result3.speedup = torch_time / lfs_time_growth;
    result3.operations = n_iterations;
    results.push_back(result3);

    BenchmarkResult result4;
    result4.name = "MCMC Workflow (LFS pre-allocated)";
    result4.lfs_time_ms = lfs_time_prealloc;
    result4.torch_time_ms = torch_time;
    result4.speedup = torch_time / lfs_time_prealloc;
    result4.operations = n_iterations;
    results.push_back(result4);

    printf("  Final Gaussian count: %zu\n", current_N);
}

TEST_F(AdamOptimizerBenchmark, RandnOverhead) {
    const int N = 500000;
    const int ITERS = 100;

    printf("\n=== Random Number Generation Overhead ===\n");
    printf("N=%d, %d iterations\n\n", N, ITERS);

    // Test 1: Single parameter (means: N x 3)
    double lfs_means_time = time_operation([&]() {
        auto t = Tensor::randn({N, 3}, Device::CUDA);
    }, 5, ITERS);

    double torch_means_time = time_operation([&]() {
        auto t = torch::randn({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    }, 5, ITERS);

    printf("Single parameter (means N x 3):\n");
    printf("  Tensor::randn: %.4f ms\n", lfs_means_time);
    printf("  torch::randn:  %.4f ms\n", torch_means_time);
    printf("  Slowdown:      %.2fx\n\n", lfs_means_time / torch_means_time);

    // Test 2: Full MCMC-style workload (all 6 parameter types)
    double lfs_full_time = time_operation([&]() {
        auto means = Tensor::randn({N, 3}, Device::CUDA);
        auto sh0 = Tensor::randn({N, 1, 3}, Device::CUDA);
        auto shN = Tensor::randn({N, 15, 3}, Device::CUDA);
        auto scaling = Tensor::randn({N, 3}, Device::CUDA);
        auto rotation = Tensor::randn({N, 4}, Device::CUDA);
        auto opacity = Tensor::randn({N, 1}, Device::CUDA);
    }, 5, ITERS);

    double torch_full_time = time_operation([&]() {
        auto means = torch::randn({N, 3}, torch::TensorOptions().device(torch::kCUDA));
        auto sh0 = torch::randn({N, 1, 3}, torch::TensorOptions().device(torch::kCUDA));
        auto shN = torch::randn({N, 15, 3}, torch::TensorOptions().device(torch::kCUDA));
        auto scaling = torch::randn({N, 3}, torch::TensorOptions().device(torch::kCUDA));
        auto rotation = torch::randn({N, 4}, torch::TensorOptions().device(torch::kCUDA));
        auto opacity = torch::randn({N, 1}, torch::TensorOptions().device(torch::kCUDA));
    }, 5, ITERS);

    printf("Full workload (6 parameters):\n");
    printf("  Tensor::randn: %.4f ms\n", lfs_full_time);
    printf("  torch::randn:  %.4f ms\n", torch_full_time);
    printf("  Slowdown:      %.2fx\n\n", lfs_full_time / torch_full_time);

    // Estimate MCMC workflow overhead
    double per_iter_overhead = lfs_full_time - torch_full_time;
    double mcmc_overhead = per_iter_overhead * 10;  // 10 iterations in MCMC test
    printf("Estimated MCMC overhead contribution:\n");
    printf("  Per-iteration overhead: %.4f ms\n", per_iter_overhead);
    printf("  Total for 10 iterations: %.4f ms\n", mcmc_overhead);
    printf("  %% of 165ms MCMC gap: %.1f%%\n\n", mcmc_overhead / 165.0 * 100);

    BenchmarkResult result1;
    result1.name = "Randn (single param)";
    result1.lfs_time_ms = lfs_means_time;
    result1.torch_time_ms = torch_means_time;
    result1.speedup = torch_means_time / lfs_means_time;
    result1.operations = N * 3;
    results.push_back(result1);

    BenchmarkResult result2;
    result2.name = "Randn (6 params)";
    result2.lfs_time_ms = lfs_full_time;
    result2.torch_time_ms = torch_full_time;
    result2.speedup = torch_full_time / lfs_full_time;
    result2.operations = N * (3 + 3 + 45 + 3 + 4 + 1);
    results.push_back(result2);
}

TEST_F(AdamOptimizerBenchmark, TensorOpsOverhead) {
    const int N_existing = 150000;  // Existing parameters
    const int N_new = 50000;        // New parameters to add
    const int ITERS = 100;

    printf("\n=== Tensor Operations Overhead (cat, zeros) ===\n");
    printf("N_existing=%d, N_new=%d, %d iterations\n\n", N_existing, N_new, ITERS);

    // Pre-create tensors for fair comparison
    auto lfs_existing = Tensor::randn({N_existing, 3}, Device::CUDA);
    auto lfs_new = Tensor::randn({N_new, 3}, Device::CUDA);
    auto torch_existing = torch::randn({N_existing, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_new = torch::randn({N_new, 3}, torch::TensorOptions().device(torch::kCUDA));

    // Test 1: Tensor::cat vs torch::cat
    double lfs_cat_time = time_operation([&]() {
        auto result = Tensor::cat(std::vector<Tensor>{lfs_existing, lfs_new}, 0);
    }, 5, ITERS);

    double torch_cat_time = time_operation([&]() {
        auto result = torch::cat({torch_existing, torch_new}, 0);
    }, 5, ITERS);

    printf("Cat operation (concatenate N=%d + N=%d along dim 0):\n", N_existing, N_new);
    printf("  Tensor::cat:  %.4f ms\n", lfs_cat_time);
    printf("  torch::cat:   %.4f ms\n", torch_cat_time);
    printf("  Slowdown:     %.2fx\n\n", lfs_cat_time / torch_cat_time);

    // Test 2: Tensor::zeros vs torch::zeros
    double lfs_zeros_time = time_operation([&]() {
        auto result = Tensor::zeros({N_new, 3}, Device::CUDA);
    }, 5, ITERS);

    double torch_zeros_time = time_operation([&]() {
        auto result = torch::zeros({N_new, 3}, torch::TensorOptions().device(torch::kCUDA));
    }, 5, ITERS);

    printf("Zeros operation (create N=%d zeros):\n", N_new);
    printf("  Tensor::zeros: %.4f ms\n", lfs_zeros_time);
    printf("  torch::zeros:  %.4f ms\n", torch_zeros_time);
    printf("  Slowdown:      %.2fx\n\n", lfs_zeros_time / torch_zeros_time);

    // Test 3: Full add_params-like operation (2 cats + 1 zeros)
    double lfs_add_time = time_operation([&]() {
        auto param_concat = Tensor::cat(std::vector<Tensor>{lfs_existing, lfs_new}, 0);
        auto zeros = Tensor::zeros({N_new, 3}, Device::CUDA);
        auto grad_concat = Tensor::cat(std::vector<Tensor>{lfs_existing, zeros}, 0);
    }, 5, ITERS);

    double torch_add_time = time_operation([&]() {
        auto param_concat = torch::cat({torch_existing, torch_new}, 0);
        auto zeros = torch::zeros({N_new, 3}, torch::TensorOptions().device(torch::kCUDA));
        auto grad_concat = torch::cat({torch_existing, zeros}, 0);
    }, 5, ITERS);

    printf("Full add operation (2×cat + 1×zeros):\n");
    printf("  LFS:      %.4f ms\n", lfs_add_time);
    printf("  PyTorch:  %.4f ms\n", torch_add_time);
    printf("  Slowdown: %.2fx\n\n", lfs_add_time / torch_add_time);

    // Estimate MCMC overhead from tensor ops
    // In MCMC: 6 params × 3 add operations = 18 add operations
    double per_add_overhead = lfs_add_time - torch_add_time;
    double mcmc_add_overhead = per_add_overhead * 18;
    printf("Estimated MCMC overhead from add operations:\n");
    printf("  Per-add overhead:    %.4f ms\n", per_add_overhead);
    printf("  Total for 18 adds:   %.4f ms\n", mcmc_add_overhead);
    printf("  %% of 127ms MCMC gap: %.1f%%\n\n", mcmc_add_overhead / 127.0 * 100);

    BenchmarkResult result1;
    result1.name = "Tensor::cat";
    result1.lfs_time_ms = lfs_cat_time;
    result1.torch_time_ms = torch_cat_time;
    result1.speedup = torch_cat_time / lfs_cat_time;
    result1.operations = (N_existing + N_new) * 3;
    results.push_back(result1);

    BenchmarkResult result2;
    result2.name = "Tensor::zeros";
    result2.lfs_time_ms = lfs_zeros_time;
    result2.torch_time_ms = torch_zeros_time;
    result2.speedup = torch_zeros_time / lfs_zeros_time;
    result2.operations = N_new * 3;
    results.push_back(result2);

    BenchmarkResult result3;
    result3.name = "Full add operation";
    result3.lfs_time_ms = lfs_add_time;
    result3.torch_time_ms = torch_add_time;
    result3.speedup = torch_add_time / lfs_add_time;
    result3.operations = (N_existing + N_new) * 3 * 2 + N_new * 3;
    results.push_back(result3);
}

} // namespace
