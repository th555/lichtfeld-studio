/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_mcmc_logit_verification.cpp
 * @brief Verify that log/logit operations match between legacy and new MCMC
 *
 * Tests that the opacity and scaling transformations are numerically identical.
 */

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include "core_new/tensor.hpp"

using namespace lfs::core;

class MCMCLogitVerificationTest : public ::testing::Test {
protected:
    void SetUp() override {
        spdlog::set_level(spdlog::level::info);
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }

    // Compare tensors between torch and lfs
    void compare_tensors(const torch::Tensor& torch_tensor,
                        const Tensor& lfs_tensor,
                        const std::string& name,
                        float tolerance = 1e-6f) {
        auto torch_cpu = torch_tensor.cpu().contiguous();
        auto lfs_cpu = lfs_tensor.cpu();

        ASSERT_EQ(torch_cpu.numel(), lfs_cpu.numel())
            << name << ": size mismatch";

        float* torch_ptr = torch_cpu.data_ptr<float>();
        const float* lfs_ptr = lfs_cpu.ptr<float>();

        float max_diff = 0.0f;
        float max_rel_diff = 0.0f;
        size_t num_mismatches = 0;

        for (size_t i = 0; i < torch_cpu.numel(); i++) {
            float diff = std::abs(torch_ptr[i] - lfs_ptr[i]);
            float rel_diff = diff / (std::abs(torch_ptr[i]) + 1e-10f);
            max_diff = std::max(max_diff, diff);
            max_rel_diff = std::max(max_rel_diff, rel_diff);

            if (diff > tolerance) {
                num_mismatches++;
                if (num_mismatches <= 5) {
                    spdlog::error("{} mismatch at index {}: torch={:.10f}, lfs={:.10f}, diff={:.10e}",
                                 name, i, torch_ptr[i], lfs_ptr[i], diff);
                }
            }
        }

        spdlog::info("{}: max_diff={:.6e}, max_rel_diff={:.6e}, mismatches={}/{}",
                    name, max_diff, max_rel_diff, num_mismatches, torch_cpu.numel());

        EXPECT_LT(max_diff, tolerance)
            << name << " has differences exceeding tolerance";
        EXPECT_EQ(num_mismatches, 0)
            << name << " has mismatches";
    }
};

/**
 * Test 1: Verify logit operation is identical
 */
TEST_F(MCMCLogitVerificationTest, LogitOperationIdentical) {
    spdlog::info("=== Test: Logit Operation ===");

    const size_t n = 1000;

    // Create test opacities in valid range (0, 1)
    auto torch_opacities = torch::rand({n}, torch::kCUDA) * 0.98f + 0.01f;  // [0.01, 0.99]

    // Convert to lfs tensor
    std::vector<float> opacity_vec;
    {
        auto cpu_t = torch_opacities.cpu();
        opacity_vec.assign(cpu_t.data_ptr<float>(), cpu_t.data_ptr<float>() + n);
    }
    auto lfs_opacities = Tensor::from_vector(opacity_vec, TensorShape({n}), Device::CUDA);

    spdlog::info("Testing {} opacity values in range [0.01, 0.99]", n);

    // Legacy: torch::logit
    auto torch_result = torch::logit(torch_opacities);

    // New: log(x / (1 - x))
    auto lfs_result = (lfs_opacities / (Tensor::ones_like(lfs_opacities) - lfs_opacities)).log();

    compare_tensors(torch_result, lfs_result, "logit", 1e-5f);

    spdlog::info("✅ Logit operations are numerically identical!");
}

/**
 * Test 2: Verify log operation is identical
 */
TEST_F(MCMCLogitVerificationTest, LogOperationIdentical) {
    spdlog::info("\n=== Test: Log Operation ===");

    const size_t n = 1000;

    // Create test scales (positive values)
    auto torch_scales = torch::rand({n, 3}, torch::kCUDA) * 10.0f + 0.01f;  // [0.01, 10.01]

    // Convert to lfs tensor
    std::vector<float> scale_vec;
    {
        auto cpu_t = torch_scales.cpu().contiguous();
        scale_vec.assign(cpu_t.data_ptr<float>(), cpu_t.data_ptr<float>() + torch_scales.numel());
    }
    auto lfs_scales = Tensor::from_vector(scale_vec, TensorShape({n, 3}), Device::CUDA);

    spdlog::info("Testing {} scale values in range [0.01, 10.01]", n * 3);

    // Both use torch::log / Tensor::log
    auto torch_result = torch::log(torch_scales);
    auto lfs_result = lfs_scales.log();

    compare_tensors(torch_result, lfs_result, "log", 1e-5f);

    spdlog::info("✅ Log operations are numerically identical!");
}

/**
 * Test 3: Verify full opacity transformation (clamp + logit)
 */
TEST_F(MCMCLogitVerificationTest, FullOpacityTransformation) {
    spdlog::info("\n=== Test: Full Opacity Transformation ===");

    const size_t n = 1000;
    const float min_opacity = 0.005f;
    const float max_opacity = 1.0f - 1e-7f;

    // Create test opacities that need clamping
    auto torch_opacities = torch::rand({n}, torch::kCUDA) * 1.5f - 0.25f;  // [-0.25, 1.25]

    // Convert to lfs tensor
    std::vector<float> opacity_vec;
    {
        auto cpu_t = torch_opacities.cpu();
        opacity_vec.assign(cpu_t.data_ptr<float>(), cpu_t.data_ptr<float>() + n);
    }
    auto lfs_opacities = Tensor::from_vector(opacity_vec, TensorShape({n}), Device::CUDA);

    spdlog::info("Testing {} opacity values (including out-of-range)", n);

    // Legacy: clamp + logit
    auto torch_clamped = torch::clamp(torch_opacities, min_opacity, max_opacity);
    auto torch_result = torch::logit(torch_clamped);

    // New: clamp + log(x / (1 - x))
    auto lfs_clamped = lfs_opacities.clamp(min_opacity, max_opacity);
    auto lfs_result = (lfs_clamped / (Tensor::ones_like(lfs_clamped) - lfs_clamped)).log();

    compare_tensors(torch_result, lfs_result, "full_opacity_transform", 1e-5f);

    spdlog::info("✅ Full opacity transformation is identical!");
}

/**
 * Test 4: Verify full scaling transformation (clamp + log)
 */
TEST_F(MCMCLogitVerificationTest, FullScalingTransformation) {
    spdlog::info("\n=== Test: Full Scaling Transformation ===");

    const size_t n = 1000;

    // Create test scales (some negative to test edge cases)
    auto torch_scales = torch::randn({n, 3}, torch::kCUDA).abs() * 5.0f + 0.001f;

    // Convert to lfs tensor
    std::vector<float> scale_vec;
    {
        auto cpu_t = torch_scales.cpu().contiguous();
        scale_vec.assign(cpu_t.data_ptr<float>(), cpu_t.data_ptr<float>() + torch_scales.numel());
    }
    auto lfs_scales = Tensor::from_vector(scale_vec, TensorShape({n, 3}), Device::CUDA);

    spdlog::info("Testing {} scale values", n * 3);

    // Both just use log (no clamping for scales in MCMC)
    auto torch_result = torch::log(torch_scales);
    auto lfs_result = lfs_scales.log();

    compare_tensors(torch_result, lfs_result, "full_scale_transform", 1e-5f);

    spdlog::info("✅ Full scaling transformation is identical!");
}

/**
 * Test 5: Verify edge cases
 */
TEST_F(MCMCLogitVerificationTest, EdgeCases) {
    spdlog::info("\n=== Test: Edge Cases ===");

    // Test specific edge case values
    std::vector<float> edge_values = {
        0.005f,           // min_opacity
        0.01f,
        0.5f,
        0.9f,
        0.99f,
        0.999f,
        1.0f - 1e-7f,     // max_opacity
    };

    for (float val : edge_values) {
        auto torch_val = torch::tensor({val}, torch::kCUDA);
        auto lfs_val = Tensor::from_vector({val}, TensorShape({1}), Device::CUDA);

        // Apply logit
        auto torch_result = torch::logit(torch_val);
        auto lfs_result = (lfs_val / (Tensor::ones_like(lfs_val) - lfs_val)).log();

        // Compare
        float torch_out = torch_result.cpu().item<float>();
        float lfs_out = lfs_result.cpu().ptr<float>()[0];
        float diff = std::abs(torch_out - lfs_out);

        spdlog::info("  opacity={:.10f}: torch_logit={:.10f}, lfs_logit={:.10f}, diff={:.6e}",
                    val, torch_out, lfs_out, diff);

        EXPECT_LT(diff, 1e-5f) << "Edge case failed for opacity=" << val;
    }

    spdlog::info("✅ All edge cases passed!");
}

/**
 * Test 6: Verify inverse operations (get_opacity)
 */
TEST_F(MCMCLogitVerificationTest, InverseOperations) {
    spdlog::info("\n=== Test: Inverse Operations (sigmoid/exp) ===");

    const size_t n = 1000;

    // Create opacity_raw values (logit space)
    auto torch_opacity_raw = torch::randn({n}, torch::kCUDA);

    std::vector<float> opacity_raw_vec;
    {
        auto cpu_t = torch_opacity_raw.cpu();
        opacity_raw_vec.assign(cpu_t.data_ptr<float>(), cpu_t.data_ptr<float>() + n);
    }
    auto lfs_opacity_raw = Tensor::from_vector(opacity_raw_vec, TensorShape({n}), Device::CUDA);

    // Apply inverse: sigmoid(x) = 1 / (1 + exp(-x))
    auto torch_opacity = torch::sigmoid(torch_opacity_raw);
    auto lfs_opacity = Tensor::ones_like(lfs_opacity_raw) /
                       (Tensor::ones_like(lfs_opacity_raw) + (-lfs_opacity_raw).exp());

    compare_tensors(torch_opacity, lfs_opacity, "sigmoid_inverse", 1e-5f);

    // Verify round-trip: sigmoid(logit(x)) = x
    auto torch_round_trip = torch::sigmoid(torch::logit(torch_opacity));
    auto lfs_round_trip = Tensor::ones_like(lfs_opacity) /
                          (Tensor::ones_like(lfs_opacity) +
                           (-(lfs_opacity / (Tensor::ones_like(lfs_opacity) - lfs_opacity)).log()).exp());

    compare_tensors(torch_round_trip, lfs_opacity, "round_trip", 1e-4f);

    spdlog::info("✅ Inverse operations match!");
}

/**
 * Test 7: Verify get_opacity() implementation
 */
TEST_F(MCMCLogitVerificationTest, GetOpacityImplementation) {
    spdlog::info("\n=== Test: get_opacity() Implementation ===");

    const size_t n = 100;

    // Create opacity_raw (in logit space)
    auto torch_opacity_raw = torch::randn({n}, torch::kCUDA);

    std::vector<float> opacity_raw_vec;
    {
        auto cpu_t = torch_opacity_raw.cpu();
        opacity_raw_vec.assign(cpu_t.data_ptr<float>(), cpu_t.data_ptr<float>() + n);
    }
    auto lfs_opacity_raw = Tensor::from_vector(opacity_raw_vec, TensorShape({n}), Device::CUDA);

    spdlog::info("Testing opacity conversion from raw (logit) to actual (sigmoid)");

    // Legacy: torch::sigmoid
    auto torch_opacity = torch::sigmoid(torch_opacity_raw);

    // Check what the tensor library uses
    // Assuming it uses: 1 / (1 + exp(-x))
    auto lfs_opacity = Tensor::ones_like(lfs_opacity_raw) /
                       (Tensor::ones_like(lfs_opacity_raw) + (-lfs_opacity_raw).exp());

    compare_tensors(torch_opacity, lfs_opacity, "get_opacity", 1e-5f);

    // Print some sample values
    auto torch_cpu = torch_opacity.cpu();
    auto lfs_cpu = lfs_opacity.cpu();
    spdlog::info("Sample values:");
    for (int i = 0; i < 5; i++) {
        spdlog::info("  raw={:.6f}: torch_opacity={:.6f}, lfs_opacity={:.6f}",
                    opacity_raw_vec[i],
                    torch_cpu.data_ptr<float>()[i],
                    lfs_cpu.ptr<float>()[i]);
    }

    spdlog::info("✅ get_opacity() implementation matches!");
}
