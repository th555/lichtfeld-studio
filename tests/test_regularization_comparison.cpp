/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_regularization_comparison.cpp
 * @brief Comprehensive comparison of opacity and scale regularization between legacy and new implementations
 *
 * Tests both single iteration and multi-iteration scenarios to ensure:
 * 1. Loss values match between implementations
 * 2. Gradients match between implementations
 * 3. Behavior remains consistent over multiple iterations with gradient descent
 */

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>
#include <cmath>

// Legacy regularization
#include "kernels/regularization.cuh"

// New regularization
#include "training_new/losses/regularization.hpp"
#include "core_new/tensor.hpp"

using namespace lfs::core;
using namespace lfs::training::losses;

class RegularizationComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        spdlog::set_level(spdlog::level::info);

        // Set random seeds for reproducibility
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }

    // Helper: Compare scalar values
    void compareScalar(float legacy_val, float new_val, const std::string& name, float tolerance = 1e-5f) {
        float diff = std::abs(legacy_val - new_val);
        spdlog::info("{}: legacy={:.8f}, new={:.8f}, diff={:.8e}", name, legacy_val, new_val, diff);
        EXPECT_LT(diff, tolerance) << name << " values differ";
    }

    // Helper: Compare tensor values between torch and lfs tensors
    void compareTensors(const torch::Tensor& legacy_tensor,
                       const Tensor& new_tensor,
                       const std::string& name,
                       float tolerance = 1e-5f) {
        // Move to CPU for comparison
        auto legacy_cpu = legacy_tensor.cpu().contiguous();
        auto new_cpu = new_tensor.cpu().contiguous();

        // Check shapes match
        ASSERT_EQ(legacy_cpu.numel(), new_cpu.numel())
            << name << " numel mismatch";

        // Compare values
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
                if (num_mismatches <= 5) {  // Print first 5 mismatches
                    spdlog::warn("{} mismatch at index {}: legacy={:.8f}, new={:.8f}, diff={:.8e}",
                                name, i, legacy_ptr[i], new_ptr[i], diff);
                }
            }
        }

        mean_diff /= numel;

        spdlog::info("{}: max_diff={:.8e}, mean_diff={:.8e}, mismatches={}/{} ({:.2f}%)",
                    name, max_diff, mean_diff, num_mismatches, numel,
                    100.0f * num_mismatches / numel);

        EXPECT_LT(max_diff, tolerance) << name << " has differences exceeding tolerance";
        EXPECT_EQ(num_mismatches, 0) << name << " has mismatches";
    }

    // Helper: Create identical test data in both formats
    std::pair<torch::Tensor, Tensor> createTestData(std::vector<int64_t> shape, float min_val = -2.0f, float max_val = 2.0f) {
        // Create torch tensor
        auto legacy_tensor = torch::rand(shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                                * (max_val - min_val) + min_val;
        legacy_tensor.requires_grad_(true);

        // Copy to lfs tensor - create tensor on CUDA with same data
        auto legacy_cpu = legacy_tensor.cpu().contiguous();
        const float* legacy_data = legacy_cpu.data_ptr<float>();

        // Create CPU tensor from blob, then copy to CUDA (which allocates and copies)
        Tensor new_tensor_cpu;
        if (shape.size() == 1) {
            new_tensor_cpu = Tensor::from_blob(const_cast<float*>(legacy_data),
                                          {static_cast<size_t>(shape[0])},
                                          Device::CPU, DataType::Float32);
        } else if (shape.size() == 2) {
            new_tensor_cpu = Tensor::from_blob(const_cast<float*>(legacy_data),
                                          {static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1])},
                                          Device::CPU, DataType::Float32);
        } else {
            throw std::runtime_error("Unsupported shape dimensionality");
        }

        // Copy to CUDA - this allocates new memory and copies the data
        auto new_tensor = new_tensor_cpu.cuda();

        return std::make_pair(legacy_tensor, new_tensor);
    }
};

// ============================================================================
// Opacity Regularization Tests
// ============================================================================

TEST_F(RegularizationComparisonTest, OpacityRegularization_SingleIteration) {
    spdlog::info("=== Testing Opacity Regularization - Single Iteration ===");

    constexpr size_t N = 10000;
    constexpr float weight = 0.01f;

    // Create identical test data - opacity is 2D [N, 1] for legacy
    auto [legacy_opacity, new_opacity] = createTestData({static_cast<int64_t>(N), 1});

    // Allocate gradient tensor for new implementation
    auto new_opacity_grad = Tensor::zeros_like(new_opacity);

    spdlog::info("=== Running Legacy Implementation ===");
    float legacy_loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
        legacy_opacity, weight);
    auto legacy_grad = legacy_opacity.grad().clone();  // Clone to preserve

    spdlog::info("=== Running New Implementation ===");
    OpacityRegularization::Params params{.weight = weight};
    auto new_loss_result = OpacityRegularization::forward(new_opacity, new_opacity_grad, params);
    ASSERT_TRUE(new_loss_result.has_value()) << "New implementation failed: " << new_loss_result.error();

    float new_loss = new_loss_result.value().item();

    spdlog::info("=== Comparing Results ===");
    compareScalar(legacy_loss, new_loss, "Opacity Regularization Loss");
    compareTensors(legacy_grad, new_opacity_grad, "Opacity Regularization Gradient");
}

TEST_F(RegularizationComparisonTest, OpacityRegularization_MultipleIterations) {
    spdlog::info("=== Testing Opacity Regularization - Multiple Iterations ===");

    constexpr size_t N = 10000;
    constexpr float weight = 0.01f;
    constexpr int num_iterations = 50;
    constexpr float learning_rate = 0.1f;

    // Create identical test data - opacity is 2D [N, 1] for legacy
    auto [legacy_opacity, new_opacity] = createTestData({static_cast<int64_t>(N), 1});

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Zero gradients
        if (legacy_opacity.grad().defined()) {
            legacy_opacity.grad().zero_();
        }
        auto new_opacity_grad = Tensor::zeros_like(new_opacity);

        // Compute loss and gradients
        float legacy_loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
            legacy_opacity, weight);

        OpacityRegularization::Params params{.weight = weight};
        auto new_loss_result = OpacityRegularization::forward(new_opacity, new_opacity_grad, params);
        ASSERT_TRUE(new_loss_result.has_value()) << "Iteration " << iter << " failed";
        float new_loss = new_loss_result.value().item();

        // Compare loss and gradients
        if (iter % 10 == 0) {
            spdlog::info("Iteration {}: legacy_loss={:.8f}, new_loss={:.8f}", iter, legacy_loss, new_loss);
        }

        compareScalar(legacy_loss, new_loss, "Loss (iter " + std::to_string(iter) + ")", 1e-4f);
        compareTensors(legacy_opacity.grad(), new_opacity_grad,
                      "Gradient (iter " + std::to_string(iter) + ")", 1e-4f);

        // Manual gradient descent update
        {
            torch::NoGradGuard no_grad;
            legacy_opacity.sub_(legacy_opacity.grad() * learning_rate);
        }

        new_opacity = new_opacity - new_opacity_grad * learning_rate;

        // Verify parameters still match after update
        auto legacy_cpu = legacy_opacity.cpu().contiguous();
        auto new_cpu = new_opacity.cpu().contiguous();
        const float* legacy_ptr = legacy_cpu.data_ptr<float>();
        const float* new_ptr = new_cpu.ptr<float>();

        float max_param_diff = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            max_param_diff = std::max(max_param_diff, std::abs(legacy_ptr[i] - new_ptr[i]));
        }

        if (iter % 10 == 0) {
            spdlog::info("Iteration {}: max parameter difference = {:.8e}", iter, max_param_diff);
        }
        EXPECT_LT(max_param_diff, 1e-3f) << "Parameters diverged at iteration " << iter;
    }

    spdlog::info("=== Multi-iteration test passed ===");
}

// ============================================================================
// Scale Regularization Tests
// ============================================================================

TEST_F(RegularizationComparisonTest, ScaleRegularization_SingleIteration) {
    spdlog::info("=== Testing Scale Regularization - Single Iteration ===");

    constexpr size_t N = 10000;
    constexpr size_t D = 3;  // 3D scaling
    constexpr float weight = 0.01f;

    // Create identical test data
    auto [legacy_scaling, new_scaling] = createTestData({static_cast<int64_t>(N), static_cast<int64_t>(D)});

    // Allocate gradient tensor for new implementation
    auto new_scaling_grad = Tensor::zeros_like(new_scaling);

    spdlog::info("=== Running Legacy Implementation ===");
    float legacy_loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
        legacy_scaling, weight);
    auto legacy_grad = legacy_scaling.grad().clone();

    spdlog::info("=== Running New Implementation ===");
    ScaleRegularization::Params params{.weight = weight};
    auto new_loss_result = ScaleRegularization::forward(new_scaling, new_scaling_grad, params);
    ASSERT_TRUE(new_loss_result.has_value()) << "New implementation failed: " << new_loss_result.error();

    float new_loss = new_loss_result.value().item();

    spdlog::info("=== Comparing Results ===");
    compareScalar(legacy_loss, new_loss, "Scale Regularization Loss");
    compareTensors(legacy_grad, new_scaling_grad, "Scale Regularization Gradient");
}

TEST_F(RegularizationComparisonTest, ScaleRegularization_MultipleIterations) {
    spdlog::info("=== Testing Scale Regularization - Multiple Iterations ===");

    constexpr size_t N = 10000;
    constexpr size_t D = 3;
    constexpr float weight = 0.01f;
    constexpr int num_iterations = 50;
    constexpr float learning_rate = 0.1f;

    // Create identical test data
    auto [legacy_scaling, new_scaling] = createTestData({static_cast<int64_t>(N), static_cast<int64_t>(D)});

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Zero gradients
        if (legacy_scaling.grad().defined()) {
            legacy_scaling.grad().zero_();
        }
        auto new_scaling_grad = Tensor::zeros_like(new_scaling);

        // Compute loss and gradients
        float legacy_loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
            legacy_scaling, weight);

        ScaleRegularization::Params params{.weight = weight};
        auto new_loss_result = ScaleRegularization::forward(new_scaling, new_scaling_grad, params);
        ASSERT_TRUE(new_loss_result.has_value()) << "Iteration " << iter << " failed";
        float new_loss = new_loss_result.value().item();

        // Compare loss and gradients
        if (iter % 10 == 0) {
            spdlog::info("Iteration {}: legacy_loss={:.8f}, new_loss={:.8f}", iter, legacy_loss, new_loss);
        }

        compareScalar(legacy_loss, new_loss, "Loss (iter " + std::to_string(iter) + ")", 1e-4f);
        compareTensors(legacy_scaling.grad(), new_scaling_grad,
                      "Gradient (iter " + std::to_string(iter) + ")", 1e-4f);

        // Manual gradient descent update
        {
            torch::NoGradGuard no_grad;
            legacy_scaling.sub_(legacy_scaling.grad() * learning_rate);
        }

        new_scaling = new_scaling - new_scaling_grad * learning_rate;

        // Verify parameters still match after update
        auto legacy_cpu = legacy_scaling.cpu().contiguous();
        auto new_cpu = new_scaling.cpu().contiguous();
        const float* legacy_ptr = legacy_cpu.data_ptr<float>();
        const float* new_ptr = new_cpu.ptr<float>();

        float max_param_diff = 0.0f;
        size_t numel = N * D;
        for (size_t i = 0; i < numel; ++i) {
            max_param_diff = std::max(max_param_diff, std::abs(legacy_ptr[i] - new_ptr[i]));
        }

        if (iter % 10 == 0) {
            spdlog::info("Iteration {}: max parameter difference = {:.8e}", iter, max_param_diff);
        }
        EXPECT_LT(max_param_diff, 1e-3f) << "Parameters diverged at iteration " << iter;
    }

    spdlog::info("=== Multi-iteration test passed ===");
}

// ============================================================================
// Combined Regularization Test
// ============================================================================

TEST_F(RegularizationComparisonTest, CombinedRegularization_MultipleIterations) {
    spdlog::info("=== Testing Combined Opacity + Scale Regularization ===");

    constexpr size_t N = 10000;
    constexpr size_t D = 3;
    constexpr float opacity_weight = 0.01f;
    constexpr float scale_weight = 0.01f;
    constexpr int num_iterations = 50;
    constexpr float learning_rate = 0.1f;

    // Create identical test data for both opacity and scaling - opacity is 2D [N, 1]
    auto [legacy_opacity, new_opacity] = createTestData({static_cast<int64_t>(N), 1});
    auto [legacy_scaling, new_scaling] = createTestData({static_cast<int64_t>(N), static_cast<int64_t>(D)});

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Zero gradients
        if (legacy_opacity.grad().defined()) {
            legacy_opacity.grad().zero_();
        }
        if (legacy_scaling.grad().defined()) {
            legacy_scaling.grad().zero_();
        }
        auto new_opacity_grad = Tensor::zeros_like(new_opacity);
        auto new_scaling_grad = Tensor::zeros_like(new_scaling);

        // Compute opacity regularization
        float legacy_opacity_loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
            legacy_opacity, opacity_weight);

        OpacityRegularization::Params opacity_params{.weight = opacity_weight};
        auto new_opacity_loss_result = OpacityRegularization::forward(new_opacity, new_opacity_grad, opacity_params);
        ASSERT_TRUE(new_opacity_loss_result.has_value()) << "Opacity regularization failed at iteration " << iter;
        float new_opacity_loss = new_opacity_loss_result.value().item();

        // Compute scale regularization
        float legacy_scale_loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
            legacy_scaling, scale_weight);

        ScaleRegularization::Params scale_params{.weight = scale_weight};
        auto new_scale_loss_result = ScaleRegularization::forward(new_scaling, new_scaling_grad, scale_params);
        ASSERT_TRUE(new_scale_loss_result.has_value()) << "Scale regularization failed at iteration " << iter;
        float new_scale_loss = new_scale_loss_result.value().item();

        // Total loss
        float legacy_total_loss = legacy_opacity_loss + legacy_scale_loss;
        float new_total_loss = new_opacity_loss + new_scale_loss;

        if (iter % 10 == 0) {
            spdlog::info("Iteration {}: legacy_total={:.8f}, new_total={:.8f}",
                        iter, legacy_total_loss, new_total_loss);
            spdlog::info("  Opacity: legacy={:.8f}, new={:.8f}", legacy_opacity_loss, new_opacity_loss);
            spdlog::info("  Scale: legacy={:.8f}, new={:.8f}", legacy_scale_loss, new_scale_loss);
        }

        // Compare losses
        compareScalar(legacy_opacity_loss, new_opacity_loss,
                     "Opacity Loss (iter " + std::to_string(iter) + ")", 1e-4f);
        compareScalar(legacy_scale_loss, new_scale_loss,
                     "Scale Loss (iter " + std::to_string(iter) + ")", 1e-4f);
        compareScalar(legacy_total_loss, new_total_loss,
                     "Total Loss (iter " + std::to_string(iter) + ")", 1e-4f);

        // Compare gradients
        compareTensors(legacy_opacity.grad(), new_opacity_grad,
                      "Opacity Gradient (iter " + std::to_string(iter) + ")", 1e-4f);
        compareTensors(legacy_scaling.grad(), new_scaling_grad,
                      "Scaling Gradient (iter " + std::to_string(iter) + ")", 1e-4f);

        // Manual gradient descent update
        {
            torch::NoGradGuard no_grad;
            legacy_opacity.sub_(legacy_opacity.grad() * learning_rate);
            legacy_scaling.sub_(legacy_scaling.grad() * learning_rate);
        }

        new_opacity = new_opacity - new_opacity_grad * learning_rate;
        new_scaling = new_scaling - new_scaling_grad * learning_rate;

        // Verify parameters remain synchronized
        if (iter % 10 == 0) {
            auto legacy_opacity_cpu = legacy_opacity.cpu().contiguous();
            auto new_opacity_cpu = new_opacity.cpu().contiguous();
            float max_opacity_diff = 0.0f;
            for (size_t i = 0; i < N; ++i) {
                max_opacity_diff = std::max(max_opacity_diff,
                    std::abs(legacy_opacity_cpu.data_ptr<float>()[i] - new_opacity_cpu.ptr<float>()[i]));
            }

            auto legacy_scaling_cpu = legacy_scaling.cpu().contiguous();
            auto new_scaling_cpu = new_scaling.cpu().contiguous();
            float max_scaling_diff = 0.0f;
            for (size_t i = 0; i < N * D; ++i) {
                max_scaling_diff = std::max(max_scaling_diff,
                    std::abs(legacy_scaling_cpu.data_ptr<float>()[i] - new_scaling_cpu.ptr<float>()[i]));
            }

            spdlog::info("Iteration {}: max opacity_diff={:.8e}, max scaling_diff={:.8e}",
                        iter, max_opacity_diff, max_scaling_diff);
        }
    }

    spdlog::info("=== Combined regularization test passed ===");
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_F(RegularizationComparisonTest, EdgeCases_ExtremeValues) {
    spdlog::info("=== Testing Edge Cases - Extreme Values ===");

    constexpr size_t N = 1000;
    constexpr float weight = 0.01f;

    // Test with very negative values (sigmoid → 0)
    {
        auto legacy_opacity = torch::ones({static_cast<int64_t>(N), 1},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * -10.0f;
        legacy_opacity.requires_grad_(true);
        auto new_opacity = Tensor::full({N, 1}, -10.0f, Device::CUDA);
        auto new_opacity_grad = Tensor::zeros_like(new_opacity);

        float legacy_loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
            legacy_opacity, weight);

        OpacityRegularization::Params params{.weight = weight};
        auto new_loss_result = OpacityRegularization::forward(new_opacity, new_opacity_grad, params);
        ASSERT_TRUE(new_loss_result.has_value());
        float new_loss = new_loss_result.value().item();

        compareScalar(legacy_loss, new_loss, "Extreme Negative Loss");
        compareTensors(legacy_opacity.grad(), new_opacity_grad, "Extreme Negative Gradient");
    }

    // Test with very positive values (sigmoid → 1)
    {
        auto legacy_opacity = torch::ones({static_cast<int64_t>(N), 1},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 10.0f;
        legacy_opacity.requires_grad_(true);
        auto new_opacity = Tensor::full({N, 1}, 10.0f, Device::CUDA);
        auto new_opacity_grad = Tensor::zeros_like(new_opacity);

        float legacy_loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
            legacy_opacity, weight);

        OpacityRegularization::Params params{.weight = weight};
        auto new_loss_result = OpacityRegularization::forward(new_opacity, new_opacity_grad, params);
        ASSERT_TRUE(new_loss_result.has_value());
        float new_loss = new_loss_result.value().item();

        compareScalar(legacy_loss, new_loss, "Extreme Positive Loss");
        compareTensors(legacy_opacity.grad(), new_opacity_grad, "Extreme Positive Gradient");
    }

    spdlog::info("=== Edge case tests passed ===");
}

TEST_F(RegularizationComparisonTest, EdgeCases_ZeroWeight) {
    spdlog::info("=== Testing Edge Cases - Zero Weight ===");

    constexpr size_t N = 1000;
    constexpr float weight = 0.0f;

    auto [legacy_opacity, new_opacity] = createTestData({static_cast<int64_t>(N), 1});
    auto new_opacity_grad = Tensor::zeros_like(new_opacity);

    float legacy_loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
        legacy_opacity, weight);

    OpacityRegularization::Params params{.weight = weight};
    auto new_loss_result = OpacityRegularization::forward(new_opacity, new_opacity_grad, params);
    ASSERT_TRUE(new_loss_result.has_value());
    float new_loss = new_loss_result.value().item();

    // Loss should be zero
    EXPECT_FLOAT_EQ(legacy_loss, 0.0f);
    EXPECT_FLOAT_EQ(new_loss, 0.0f);

    // Gradients should be zero (or undefined for legacy when weight=0)
    if (legacy_opacity.grad().defined()) {
        auto legacy_grad = legacy_opacity.grad();
        EXPECT_TRUE(legacy_grad.abs().max().item<float>() < 1e-7f);
    }
    EXPECT_TRUE(new_opacity_grad.abs().max().item() < 1e-7f);

    spdlog::info("=== Zero weight test passed ===");
}
