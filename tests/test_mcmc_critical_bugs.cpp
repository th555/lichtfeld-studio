/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Critical bug regression tests for MCMC training pipeline.
 *
 * This file consolidates tests for several critical bugs discovered during MCMC development:
 *
 * 1. Adam Optimizer + add_new_params Bug (src: test_add_new_params_adam_bug.cpp)
 *    - Divergence when adding new parameters after Adam steps
 *    - Optimizer state capacity/corruption issues
 *
 * 2. inject_noise Kernel Validation (src: test_inject_noise_validation.cpp)
 *    - Matrix transpose bugs in noise injection
 *    - Opacity modulation correctness
 *    - Numerical stability with extreme values
 *
 * 3. Tensor::cat Self-Reference Bug (src: test_tensor_cat_reference_bug.cpp)
 *    - Memory corruption when param = cat({param, new_values})
 *    - Progressive corruption over multiple concatenations
 *
 * 4. index_add_ Int64 Bug (src: test_tensor_index_add_bug.cpp)
 *    - Incorrect casting of int64 indices (from nonzero()) to int*
 *    - MCMC ratio counting failures
 */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "core_new/tensor.hpp"
#include "core_new/splat_data.hpp"
#include "training_new/optimizer/adam_optimizer.hpp"
#include "kernels/mcmc_kernels.hpp"
#include "Ops.h"  // gsplat reference

using namespace lfs::core;
using namespace lfs::training;
using namespace lfs::training::mcmc;

// ============================================================================
// SECTION 1: ADAM OPTIMIZER + add_new_params BUG TESTS
// ============================================================================
// These tests expose bugs where add_new_params + Adam optimization causes
// divergence due to optimizer state corruption.
// ============================================================================

TEST(MCMCCriticalBugs_Adam, ParameterDivergenceAfterAddNewParams) {
    // Create SplatData with minimal initialization
    Tensor initial_means = Tensor::arange(0.0f, 300.0f).cuda().reshape({100, 3});
    Tensor sh0 = Tensor::zeros({100, 3}, Device::CUDA);
    Tensor shN = Tensor::zeros({100, 0}, Device::CUDA);  // No higher-order SH
    Tensor scaling = Tensor::zeros({100, 3}, Device::CUDA);
    Tensor rotation = Tensor::zeros({100, 4}, Device::CUDA);
    Tensor opacity = Tensor::zeros({100, 1}, Device::CUDA);

    SplatData splat_data(0, initial_means, sh0, shN, scaling, rotation, opacity, 1.0f);
    splat_data.allocate_gradients();

    // Create Adam optimizer
    AdamConfig config;
    config.lr = 1e-3f;
    config.initial_capacity = 1000;  // Pre-allocate

    AdamOptimizer optimizer(splat_data, config);

    // Track original values to detect corruption
    auto track_cpu = splat_data.means().cpu();
    float original_first = track_cpu.ptr<float>()[0];
    float original_last = track_cpu.ptr<float>()[299];  // 100*3 - 1

    // Simulate training loop
    for (int iter = 0; iter < 10; iter++) {
        // Set some fake gradients
        auto means_grad = splat_data.means_grad();
        means_grad.fill_(0.001f);

        // Run Adam step
        optimizer.step(iter);

        // Add new Gaussians every iteration (like MCMC does)
        Tensor new_means = Tensor::ones({20, 3}, Device::CUDA) * static_cast<float>(1000 + iter * 100);
        optimizer.add_new_params(ParamType::Means, new_means);

        // Check original values haven't corrupted
        auto current_cpu = splat_data.means().cpu();
        float current_first = current_cpu.ptr<float>()[0];
        float current_last_original = current_cpu.ptr<float>()[299];

        // Allow for Adam updates, but check the values are reasonable
        EXPECT_LT(std::abs(current_first - original_first), 1.0f)
            << "Iteration " << iter << ": First value diverged from "
            << original_first << " to " << current_first;

        EXPECT_LT(std::abs(current_last_original - original_last), 1.0f)
            << "Iteration " << iter << ": Last original value diverged from "
            << original_last << " to " << current_last_original;

        // Update tracking
        original_first = current_first;
        original_last = current_last_original;
    }

    // Final size check
    EXPECT_EQ(splat_data.means().shape()[0], 100 + 10 * 20);
}

TEST(MCMCCriticalBugs_Adam, ExponentialDivergenceDetection) {
    // This test specifically checks for the exponential divergence pattern
    Tensor initial_means = Tensor::zeros({100, 3}, Device::CUDA);
    Tensor sh0 = Tensor::zeros({100, 3}, Device::CUDA);
    Tensor shN = Tensor::zeros({100, 0}, Device::CUDA);
    Tensor scaling = Tensor::zeros({100, 3}, Device::CUDA);
    Tensor rotation = Tensor::zeros({100, 4}, Device::CUDA);
    Tensor opacity = Tensor::zeros({100, 1}, Device::CUDA);

    SplatData splat_data(0, initial_means, sh0, shN, scaling, rotation, opacity, 1.0f);
    splat_data.allocate_gradients();

    AdamConfig config;
    config.lr = 1e-2f;  // Higher LR to amplify any bugs
    config.initial_capacity = 1000;

    AdamOptimizer optimizer(splat_data, config);

    std::vector<float> max_abs_values;

    for (int iter = 0; iter < 20; iter++) {
        // Small constant gradients
        auto means_grad = splat_data.means_grad();
        means_grad.fill_(0.01f);

        optimizer.step(iter);

        // Add new Gaussians
        Tensor new_means = Tensor::randn({50, 3}, Device::CUDA) * 0.1f;
        optimizer.add_new_params(ParamType::Means, new_means);

        // Track maximum absolute value
        auto current_cpu = splat_data.means().cpu();
        const float* data = current_cpu.ptr<float>();
        size_t n_elements = std::min(current_cpu.numel(), size_t(300));  // First 100 Gaussians

        float max_abs = 0.0f;
        for (size_t i = 0; i < n_elements; i++) {
            max_abs = std::max(max_abs, std::abs(data[i]));
        }
        max_abs_values.push_back(max_abs);

        // With constant small gradients and proper Adam, values should stay bounded
        EXPECT_LT(max_abs, 10.0f)
            << "Iteration " << iter << ": Values exploded to " << max_abs
            << " (exponential divergence detected!)";
    }

    // Check that growth is sub-linear (not exponential)
    if (max_abs_values.size() >= 10) {
        float early_max = max_abs_values[5];
        float late_max = max_abs_values[15];

        // Late values shouldn't be more than 3x early values (exponential would be >>10x)
        EXPECT_LT(late_max / early_max, 3.0f)
            << "Exponential growth detected: early=" << early_max
            << " late=" << late_max << " ratio=" << (late_max / early_max);
    }
}

TEST(MCMCCriticalBugs_Adam, StateCapacityCorruption) {
    // Test if the optimizer state capacity tracking causes corruption
    Tensor initial_means = Tensor::ones({50, 3}, Device::CUDA);
    Tensor sh0 = Tensor::zeros({50, 3}, Device::CUDA);
    Tensor shN = Tensor::zeros({50, 0}, Device::CUDA);
    Tensor scaling = Tensor::zeros({50, 3}, Device::CUDA);
    Tensor rotation = Tensor::zeros({50, 4}, Device::CUDA);
    Tensor opacity = Tensor::zeros({50, 1}, Device::CUDA);

    SplatData splat_data(0, initial_means, sh0, shN, scaling, rotation, opacity, 1.0f);
    splat_data.allocate_gradients();

    AdamConfig config;
    config.lr = 1e-3f;
    config.initial_capacity = 200;  // Pre-allocate for 200 Gaussians

    AdamOptimizer optimizer(splat_data, config);

    // Run several iterations with add_new_params within capacity
    for (int iter = 0; iter < 5; iter++) {
        auto means_grad = splat_data.means_grad();
        means_grad.fill_(0.001f);

        optimizer.step(iter);

        // Add 20 Gaussians (stays within 200 capacity)
        Tensor new_means = Tensor::ones({20, 3}, Device::CUDA) * 5.0f;
        optimizer.add_new_params(ParamType::Means, new_means);

        // Verify state is consistent
        auto state = optimizer.get_state(ParamType::Means);
        ASSERT_NE(state, nullptr);

        size_t expected_size = 50 + (iter + 1) * 20;
        EXPECT_EQ(state->size, expected_size)
            << "Iteration " << iter << ": State size mismatch";
        EXPECT_EQ(splat_data.means().shape()[0], expected_size)
            << "Iteration " << iter << ": Parameter size mismatch";
    }
}

// ============================================================================
// SECTION 2: INJECT_NOISE KERNEL VALIDATION TESTS
// ============================================================================
// These tests validate the new inject_noise kernel against the legacy gsplat
// implementation, catching matrix transpose and numerical precision bugs.
// ============================================================================

// Helper to compare tensors with detailed error reporting
static bool tensors_close_detailed(const Tensor& a, const torch::Tensor& b,
                            float rtol, float atol,
                            std::string test_name) {
    auto a_torch = torch::from_blob(
        const_cast<float*>(a.ptr<float>()),
        {static_cast<long>(a.numel())},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto b_flat = b.flatten();
    auto diff = (a_torch - b_flat).abs();
    auto threshold = atol + rtol * b_flat.abs();
    auto matches = (diff <= threshold);

    bool all_match = matches.all().item<bool>();

    if (!all_match) {
        auto max_diff = diff.max().item<float>();
        auto mean_diff = diff.mean().item<float>();
        auto num_mismatches = (matches == 0).sum().item<int>();

        std::cout << "\n❌ " << test_name << " FAILED:\n";
        std::cout << "   Max diff: " << max_diff << "\n";
        std::cout << "   Mean diff: " << mean_diff << "\n";
        std::cout << "   Mismatches: " << num_mismatches << "/" << a.numel() << "\n";
        std::cout << "   Tolerance: rtol=" << rtol << ", atol=" << atol << "\n";

        // Show first few mismatches
        auto diff_cpu = diff.cpu();
        auto a_cpu = a_torch.cpu();
        auto b_cpu = b_flat.cpu();
        int shown = 0;
        for (int i = 0; i < std::min<int>(10, a.numel()) && shown < 5; i++) {
            if (!matches[i].item<bool>()) {
                std::cout << "   [" << i << "] new=" << a_cpu[i].item<float>()
                         << ", legacy=" << b_cpu[i].item<float>()
                         << ", diff=" << diff_cpu[i].item<float>() << "\n";
                shown++;
            }
        }
    } else {
        auto max_diff = diff.max().item<float>();
        std::cout << "✓ " << test_name << " PASSED (max_diff=" << max_diff << ")\n";
    }

    return all_match;
}

TEST(MCMCCriticalBugs_InjectNoise, IdentityRotation_ZeroScales) {
    // This test checks basic matrix multiplication correctness
    // Identity rotation + zero scales should produce zero covariance
    const size_t N = 10;

    // Identity quaternion: [1, 0, 0, 0]
    auto raw_quats_torch = torch::zeros({static_cast<long>(N), 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    raw_quats_torch.index_put_({torch::indexing::Slice(), 0}, 1.0f);  // w = 1

    // Zero scales (log-space, so exp(0) = 1, but we use very small values)
    auto raw_scales_torch = torch::full({static_cast<long>(N), 3}, -10.0f,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // High opacity (sigmoid will be close to 1)
    auto raw_opacities_torch = torch::full({static_cast<long>(N)}, 5.0f,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Unit noise
    auto noise_torch = torch::ones({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto means_torch = torch::zeros({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    float current_lr = 0.001f;

    // Reference
    auto means_ref = means_torch.clone();
    gsplat::add_noise(raw_opacities_torch, raw_scales_torch, raw_quats_torch,
                      noise_torch, means_ref, current_lr);

    // Test
    auto means_test = means_torch.clone();
    Tensor means_test_tensor = Tensor::from_blob(
        means_test.data_ptr<float>(),
        TensorShape({N, 3}),
        Device::CUDA,
        DataType::Float32);

    launch_add_noise_kernel(
        raw_opacities_torch.data_ptr<float>(),
        raw_scales_torch.data_ptr<float>(),
        raw_quats_torch.data_ptr<float>(),
        noise_torch.data_ptr<float>(),
        means_test_tensor.ptr<float>(),
        current_lr,
        N);

    cudaDeviceSynchronize();

    EXPECT_TRUE(tensors_close_detailed(means_test_tensor, means_ref, 1e-5f, 1e-6f,
                                       "Identity rotation + zero scales"));
}

TEST(MCMCCriticalBugs_InjectNoise, AsymmetricRotation_DetectsTranspose) {
    // Use a rotation that would give DIFFERENT results if transposed
    // This catches column-major vs row-major errors
    const size_t N = 100;

    // Quaternion for 90-degree rotation around Z-axis
    auto raw_quats_torch = torch::zeros({static_cast<long>(N), 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    raw_quats_torch.index_put_({torch::indexing::Slice(), 0}, 0.707f);  // w
    raw_quats_torch.index_put_({torch::indexing::Slice(), 3}, 0.707f);  // z

    // Asymmetric scales: x != y != z
    auto raw_scales_torch = torch::zeros({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    raw_scales_torch.index_put_({torch::indexing::Slice(), 0}, std::log(2.0f));  // x scale
    raw_scales_torch.index_put_({torch::indexing::Slice(), 1}, std::log(0.5f));  // y scale
    raw_scales_torch.index_put_({torch::indexing::Slice(), 2}, std::log(1.0f));  // z scale

    auto raw_opacities_torch = torch::full({static_cast<long>(N)}, 2.0f,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Asymmetric noise: [1, 2, 3]
    auto noise_torch = torch::zeros({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    noise_torch.index_put_({torch::indexing::Slice(), 0}, 1.0f);
    noise_torch.index_put_({torch::indexing::Slice(), 1}, 2.0f);
    noise_torch.index_put_({torch::indexing::Slice(), 2}, 3.0f);

    auto means_torch = torch::zeros({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    float current_lr = 0.01f;

    // Reference
    auto means_ref = means_torch.clone();
    gsplat::add_noise(raw_opacities_torch, raw_scales_torch, raw_quats_torch,
                      noise_torch, means_ref, current_lr);

    // Test
    auto means_test = means_torch.clone();
    Tensor means_test_tensor = Tensor::from_blob(
        means_test.data_ptr<float>(),
        TensorShape({N, 3}),
        Device::CUDA,
        DataType::Float32);

    launch_add_noise_kernel(
        raw_opacities_torch.data_ptr<float>(),
        raw_scales_torch.data_ptr<float>(),
        raw_quats_torch.data_ptr<float>(),
        noise_torch.data_ptr<float>(),
        means_test_tensor.ptr<float>(),
        current_lr,
        N);

    cudaDeviceSynchronize();

    // TIGHT tolerance - any matrix order issue will fail this
    EXPECT_TRUE(tensors_close_detailed(means_test_tensor, means_ref, 1e-5f, 1e-5f,
                                       "Asymmetric rotation (transpose detection)"));
}

TEST(MCMCCriticalBugs_InjectNoise, RandomRotations_TightTolerance) {
    // Random rotations with tight tolerance to catch numerical issues
    const size_t N = 1000;

    auto raw_quats_torch = torch::randn({static_cast<long>(N), 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto raw_scales_torch = torch::randn({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;

    auto raw_opacities_torch = torch::randn({static_cast<long>(N)},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto noise_torch = torch::randn({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto means_torch = torch::randn({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 2.0f;

    float current_lr = 0.001f;

    // Reference
    auto means_ref = means_torch.clone();
    gsplat::add_noise(raw_opacities_torch, raw_scales_torch, raw_quats_torch,
                      noise_torch, means_ref, current_lr);

    // Test
    auto means_test = means_torch.clone();
    Tensor means_test_tensor = Tensor::from_blob(
        means_test.data_ptr<float>(),
        TensorShape({N, 3}),
        Device::CUDA,
        DataType::Float32);

    launch_add_noise_kernel(
        raw_opacities_torch.data_ptr<float>(),
        raw_scales_torch.data_ptr<float>(),
        raw_quats_torch.data_ptr<float>(),
        noise_torch.data_ptr<float>(),
        means_test_tensor.ptr<float>(),
        current_lr,
        N);

    cudaDeviceSynchronize();

    // Tighter tolerance than existing test (1e-2, 1e-3)
    EXPECT_TRUE(tensors_close_detailed(means_test_tensor, means_ref, 1e-4f, 1e-5f,
                                       "Random rotations (tight tolerance)"));
}

TEST(MCMCCriticalBugs_InjectNoise, ExtremeValues_Stability) {
    // Test numerical stability with extreme values
    const size_t N = 100;

    // Very high opacities (near saturation)
    auto raw_opacities_torch = torch::full({static_cast<long>(N)}, 10.0f,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Very large scales
    auto raw_scales_torch = torch::full({static_cast<long>(N), 3}, 5.0f,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Random quaternions (will be normalized)
    auto raw_quats_torch = torch::randn({static_cast<long>(N), 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 10.0f;

    auto noise_torch = torch::randn({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 5.0f;

    auto means_torch = torch::randn({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 100.0f;

    float current_lr = 1.0f;

    // Reference
    auto means_ref = means_torch.clone();
    gsplat::add_noise(raw_opacities_torch, raw_scales_torch, raw_quats_torch,
                      noise_torch, means_ref, current_lr);

    // Test
    auto means_test = means_torch.clone();
    Tensor means_test_tensor = Tensor::from_blob(
        means_test.data_ptr<float>(),
        TensorShape({N, 3}),
        Device::CUDA,
        DataType::Float32);

    launch_add_noise_kernel(
        raw_opacities_torch.data_ptr<float>(),
        raw_scales_torch.data_ptr<float>(),
        raw_quats_torch.data_ptr<float>(),
        noise_torch.data_ptr<float>(),
        means_test_tensor.ptr<float>(),
        current_lr,
        N);

    cudaDeviceSynchronize();

    // Check for NaN/Inf
    auto means_test_torch = torch::from_blob(
        const_cast<float*>(means_test_tensor.ptr<float>()),
        {static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    EXPECT_FALSE(means_test_torch.isnan().any().item<bool>()) << "New implementation produced NaN";
    EXPECT_FALSE(means_test_torch.isinf().any().item<bool>()) << "New implementation produced Inf";
    EXPECT_FALSE(means_ref.isnan().any().item<bool>()) << "Reference produced NaN";
    EXPECT_FALSE(means_ref.isinf().any().item<bool>()) << "Reference produced Inf";

    EXPECT_TRUE(tensors_close_detailed(means_test_tensor, means_ref, 1e-3f, 1e-4f,
                                       "Extreme values (stability)"));
}

TEST(MCMCCriticalBugs_InjectNoise, RealisticMCMC_PixelPerfect) {
    // Simulate realistic MCMC training conditions
    const size_t N = 5000;

    // Realistic opacities after training
    auto raw_opacities_torch = (torch::rand({static_cast<long>(N)},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) - 0.5f) * 4.0f;

    // Log-normal scale distribution
    auto raw_scales_torch = torch::randn({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 0.3f - 1.0f;

    // Normalized quaternions with small perturbations
    auto raw_quats_torch = torch::randn({static_cast<long>(N), 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 0.1f;
    raw_quats_torch.index_put_({torch::indexing::Slice(), 0},
        raw_quats_torch.index({torch::indexing::Slice(), 0}) + 1.0f);

    // Standard Gaussian noise
    auto noise_torch = torch::randn({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Scene-scale means
    auto means_torch = torch::randn({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 2.0f;

    // Realistic MCMC learning rate
    float optimizer_lr = 1e-4f;  // means_lr after decay
    float noise_lr = 5e5f;        // MCMC noise multiplier
    float current_lr = optimizer_lr * noise_lr;  // = 50.0

    // Verify inputs are IDENTICAL before testing
    auto means_ref = means_torch.clone();
    auto means_test = means_torch.clone();

    ASSERT_TRUE(torch::allclose(means_ref, means_test, 0.0, 0.0))
        << "Initial means should be identical before noise injection";

    // Reference
    gsplat::add_noise(raw_opacities_torch, raw_scales_torch, raw_quats_torch,
                      noise_torch, means_ref, current_lr);

    // Test
    Tensor means_test_tensor = Tensor::from_blob(
        means_test.data_ptr<float>(),
        TensorShape({N, 3}),
        Device::CUDA,
        DataType::Float32);

    ASSERT_EQ(means_test_tensor.ptr<float>(), means_test.data_ptr<float>())
        << "Tensor wrapper must point to same memory as LibTorch tensor";

    launch_add_noise_kernel(
        raw_opacities_torch.data_ptr<float>(),
        raw_scales_torch.data_ptr<float>(),
        raw_quats_torch.data_ptr<float>(),
        noise_torch.data_ptr<float>(),
        means_test_tensor.ptr<float>(),
        current_lr,
        N);

    cudaDeviceSynchronize();

    EXPECT_TRUE(tensors_close_detailed(means_test_tensor, means_ref, 1e-4f, 5e-4f,
                                       "Realistic MCMC (high fidelity)"));
}

TEST(MCMCCriticalBugs_InjectNoise, OpacityModulation_Correctness) {
    // Verify that opacity correctly modulates noise strength
    // NOTE: In MCMC, HIGH opacity Gaussians get LESS noise (they're well-established)
    const size_t N = 100;

    // Test with low opacity (sigmoid ≈ 0.007, op_sigmoid ≈ 0.64)
    auto raw_opacities_low = torch::full({static_cast<long>(N)}, -5.0f,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Test with high opacity (sigmoid ≈ 0.993, op_sigmoid ≈ 0)
    auto raw_opacities_high = torch::full({static_cast<long>(N)}, 5.0f,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto raw_scales_torch = torch::zeros({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto raw_quats_torch = torch::zeros({static_cast<long>(N), 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    raw_quats_torch.index_put_({torch::indexing::Slice(), 0}, 1.0f);

    auto noise_torch = torch::ones({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto means_torch = torch::zeros({static_cast<long>(N), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    float current_lr = 1.0f;

    // Test low opacity
    auto means_low_ref = means_torch.clone();
    auto means_low_test = means_torch.clone();
    Tensor means_low_tensor = Tensor::from_blob(
        means_low_test.data_ptr<float>(), TensorShape({N, 3}), Device::CUDA, DataType::Float32);

    gsplat::add_noise(raw_opacities_low, raw_scales_torch, raw_quats_torch,
                      noise_torch, means_low_ref, current_lr);
    launch_add_noise_kernel(
        raw_opacities_low.data_ptr<float>(), raw_scales_torch.data_ptr<float>(),
        raw_quats_torch.data_ptr<float>(), noise_torch.data_ptr<float>(),
        means_low_tensor.ptr<float>(), current_lr, N);

    // Test high opacity
    auto means_high_ref = means_torch.clone();
    auto means_high_test = means_torch.clone();
    Tensor means_high_tensor = Tensor::from_blob(
        means_high_test.data_ptr<float>(), TensorShape({N, 3}), Device::CUDA, DataType::Float32);

    gsplat::add_noise(raw_opacities_high, raw_scales_torch, raw_quats_torch,
                      noise_torch, means_high_ref, current_lr);
    launch_add_noise_kernel(
        raw_opacities_high.data_ptr<float>(), raw_scales_torch.data_ptr<float>(),
        raw_quats_torch.data_ptr<float>(), noise_torch.data_ptr<float>(),
        means_high_tensor.ptr<float>(), current_lr, N);

    cudaDeviceSynchronize();

    EXPECT_TRUE(tensors_close_detailed(means_low_tensor, means_low_ref, 1e-6f, 1e-7f,
                                       "Low opacity modulation"));
    EXPECT_TRUE(tensors_close_detailed(means_high_tensor, means_high_ref, 1e-6f, 1e-7f,
                                       "High opacity modulation"));

    // Verify LOW opacity produces larger changes than HIGH opacity
    auto low_change = means_low_test.abs().mean().item<float>();
    auto high_change = means_high_test.abs().mean().item<float>();
    EXPECT_GT(low_change, high_change) << "Low opacity should produce more noise than high opacity";
}

// ============================================================================
// SECTION 3: TENSOR::CAT SELF-REFERENCE BUG TESTS
// ============================================================================
// These tests expose bugs where param = Tensor::cat({param, new_values}, 0)
// causes memory corruption when param is a reference.
// ============================================================================

TEST(MCMCCriticalBugs_TensorCat, SelfReferentialCatWithReference) {
    // Simulate the optimizer pattern: param is a reference to SplatData
    Tensor original = Tensor::arange(0.0f, 10.0f).cuda().reshape({10, 1});
    Tensor& param_ref = original;

    // Verify initial state
    auto init_cpu = param_ref.cpu();
    EXPECT_FLOAT_EQ(init_cpu.ptr<float>()[0], 0.0f);
    EXPECT_FLOAT_EQ(init_cpu.ptr<float>()[9], 9.0f);
    EXPECT_EQ(param_ref.shape()[0], 10);

    // Simulate add_new_params: param = cat({param, new_values}, 0)
    Tensor new_values = Tensor::ones({5, 1}, Device::CUDA) * 100.0f;
    param_ref = Tensor::cat({param_ref, new_values}, 0);

    // Check results
    EXPECT_EQ(param_ref.shape()[0], 15);
    EXPECT_EQ(original.shape()[0], 15);  // Should be updated due to aliasing

    auto result_cpu = param_ref.cpu();

    // Original values should be preserved
    EXPECT_FLOAT_EQ(result_cpu.ptr<float>()[0], 0.0f) << "First original value corrupted";
    EXPECT_FLOAT_EQ(result_cpu.ptr<float>()[5], 5.0f) << "Middle original value corrupted";
    EXPECT_FLOAT_EQ(result_cpu.ptr<float>()[9], 9.0f) << "Last original value corrupted";

    // New values should be appended
    EXPECT_FLOAT_EQ(result_cpu.ptr<float>()[10], 100.0f) << "First new value incorrect";
    EXPECT_FLOAT_EQ(result_cpu.ptr<float>()[14], 100.0f) << "Last new value incorrect";
}

TEST(MCMCCriticalBugs_TensorCat, ProgressiveCorruptionSimulation) {
    // Simulate multiple add_new_gs calls like in training
    Tensor means = Tensor::arange(0.0f, 100.0f).cuda().reshape({100, 1});
    Tensor& means_ref = means;

    // Track values at specific indices to detect corruption
    std::vector<int> tracked_indices = {0, 10, 50, 99};
    std::vector<float> initial_values;

    auto initial_cpu = means_ref.cpu();
    for (int idx : tracked_indices) {
        initial_values.push_back(initial_cpu.ptr<float>()[idx]);
    }

    // Simulate 10 iterations of add_new_gs
    for (int iter = 0; iter < 10; iter++) {
        // Add 20 new Gaussians each iteration (like MCMC does)
        Tensor new_means = Tensor::ones({20, 1}, Device::CUDA) * static_cast<float>(1000 + iter * 100);
        means_ref = Tensor::cat({means_ref, new_means}, 0);

        // Verify original values haven't been corrupted
        auto current_cpu = means_ref.cpu();
        for (size_t i = 0; i < tracked_indices.size(); i++) {
            int idx = tracked_indices[i];
            float expected = initial_values[i];
            float actual = current_cpu.ptr<float>()[idx];

            EXPECT_FLOAT_EQ(actual, expected)
                << "Iteration " << iter << ": Value at index " << idx
                << " corrupted from " << expected << " to " << actual;
        }
    }

    // Final size check
    EXPECT_EQ(means_ref.shape()[0], 100 + 10 * 20);
}

TEST(MCMCCriticalBugs_TensorCat, WithCapacityPreallocation) {
    // Test with capacity pre-allocation (like optimizer state)
    Tensor original = Tensor::zeros({100, 3}, Device::CUDA);
    original.reserve(1000);  // Pre-allocate capacity

    Tensor& param_ref = original;

    // Verify initial values
    auto init_cpu = param_ref.cpu();
    float init_sum = 0.0f;
    for (size_t i = 0; i < 300; i++) {  // 100 * 3
        init_sum += init_cpu.ptr<float>()[i];
    }
    EXPECT_FLOAT_EQ(init_sum, 0.0f);

    // Add new values multiple times using the reference pattern
    for (int iter = 0; iter < 5; iter++) {
        Tensor new_vals = Tensor::ones({50, 3}, Device::CUDA) * static_cast<float>(iter + 1);
        param_ref = Tensor::cat({param_ref, new_vals}, 0);

        // Check that zeros are still zeros
        auto current_cpu = param_ref.cpu();
        for (size_t i = 0; i < 100; i++) {
            EXPECT_FLOAT_EQ(current_cpu.ptr<float>()[i * 3], 0.0f)
                << "Iteration " << iter << ": Original value at " << i << " corrupted";
        }
    }
}

TEST(MCMCCriticalBugs_TensorCat, MultipleReferencesSimulation) {
    // Simulate the scenario where both SplatData and Optimizer hold references
    Tensor shared_data = Tensor::arange(0.0f, 50.0f).cuda().reshape({50, 1});

    // Two references (like SplatData.means() and Optimizer.get_param())
    Tensor& splat_ref = shared_data;
    Tensor& optimizer_ref = shared_data;

    // Verify they point to the same data
    EXPECT_EQ(splat_ref.raw_ptr(), optimizer_ref.raw_ptr());

    // Do concatenation through optimizer reference
    Tensor new_data = Tensor::ones({25, 1}, Device::CUDA) * 999.0f;
    optimizer_ref = Tensor::cat({optimizer_ref, new_data}, 0);

    // Both references should see the update
    EXPECT_EQ(optimizer_ref.shape()[0], 75);
    EXPECT_EQ(splat_ref.shape()[0], 75);
    EXPECT_EQ(shared_data.shape()[0], 75);

    // Verify data integrity
    auto result_cpu = splat_ref.cpu();

    // Original values
    EXPECT_FLOAT_EQ(result_cpu.ptr<float>()[0], 0.0f);
    EXPECT_FLOAT_EQ(result_cpu.ptr<float>()[25], 25.0f);
    EXPECT_FLOAT_EQ(result_cpu.ptr<float>()[49], 49.0f);

    // New values
    EXPECT_FLOAT_EQ(result_cpu.ptr<float>()[50], 999.0f);
    EXPECT_FLOAT_EQ(result_cpu.ptr<float>()[74], 999.0f);
}

TEST(MCMCCriticalBugs_TensorCat, InPlaceVsNewAllocation) {
    // Test to see if the bug is in the in-place optimization path

    // Scenario 1: With capacity (in-place path)
    Tensor with_capacity = Tensor::ones({100, 3}, Device::CUDA);
    with_capacity.reserve(500);
    Tensor& ref_with_capacity = with_capacity;

    void* original_ptr = ref_with_capacity.raw_ptr();

    Tensor new_vals1 = Tensor::zeros({50, 3}, Device::CUDA);
    ref_with_capacity = Tensor::cat({ref_with_capacity, new_vals1}, 0);

    bool used_inplace = (ref_with_capacity.raw_ptr() == original_ptr);

    auto result1_cpu = ref_with_capacity.cpu();
    float sum1 = 0.0f;
    for (size_t i = 0; i < 300; i++) {  // First 100 * 3 elements
        sum1 += result1_cpu.ptr<float>()[i];
    }
    EXPECT_FLOAT_EQ(sum1, 300.0f) << "In-place path corrupted data (used_inplace=" << used_inplace << ")";

    // Scenario 2: Without capacity (new allocation path)
    Tensor without_capacity = Tensor::ones({100, 3}, Device::CUDA);
    Tensor& ref_without_capacity = without_capacity;

    Tensor new_vals2 = Tensor::zeros({50, 3}, Device::CUDA);
    ref_without_capacity = Tensor::cat({ref_without_capacity, new_vals2}, 0);

    auto result2_cpu = ref_without_capacity.cpu();
    float sum2 = 0.0f;
    for (size_t i = 0; i < 300; i++) {
        sum2 += result2_cpu.ptr<float>()[i];
    }
    EXPECT_FLOAT_EQ(sum2, 300.0f) << "New allocation path corrupted data";
}

// ============================================================================
// SECTION 4: INDEX_ADD_ INT64 BUG TESTS
// ============================================================================
// These tests expose bugs where int64 indices (from nonzero()) are incorrectly
// cast to int* in the index_add_ operation, causing MCMC ratio counting failures.
// ============================================================================

// Test fixture for index_add_ bug
class MCMCCriticalBugs_IndexAdd : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure deterministic behavior
        torch::manual_seed(42);
    }

    // Helper to compare LFS tensor with LibTorch tensor
    void assertTensorsEqual(const Tensor& lfs_tensor, const torch::Tensor& torch_tensor,
                           float tolerance = 1e-5f) {
        ASSERT_EQ(lfs_tensor.numel(), torch_tensor.numel());

        auto lfs_cpu = lfs_tensor.cpu();
        auto torch_cpu = torch_tensor.cpu();

        const float* lfs_data = lfs_cpu.ptr<float>();
        const float* torch_data = torch_cpu.data_ptr<float>();

        for (size_t i = 0; i < lfs_tensor.numel(); ++i) {
            EXPECT_NEAR(lfs_data[i], torch_data[i], tolerance)
                << "Mismatch at index " << i
                << ": LFS=" << lfs_data[i]
                << ", Torch=" << torch_data[i];
        }
    }
};

TEST_F(MCMCCriticalBugs_IndexAdd, Int64IndicesBug) {
    // This test exposes the bug where int64 indices are incorrectly cast to int*
    const int N = 1000;
    const int num_samples = 100;

    // Create base tensor of ones
    auto lfs_ratios = Tensor::ones({N}, Device::CUDA, DataType::Float32);
    auto torch_ratios = torch::ones({N}, torch::kFloat32).cuda();

    // Create int64 indices (this is what nonzero() returns)
    std::vector<int64_t> idx_data(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        idx_data[i] = rand() % N;
    }

    auto lfs_indices = Tensor::from_blob(idx_data.data(), {num_samples}, Device::CPU, DataType::Int64).cuda();
    auto torch_indices = torch::from_blob(idx_data.data(), {num_samples}, torch::kInt64).cuda();

    // Create source tensor of ones to add
    auto lfs_src = Tensor::ones({num_samples}, Device::CUDA, DataType::Float32);
    auto torch_src = torch::ones({num_samples}, torch::kFloat32).cuda();

    // Perform index_add_
    lfs_ratios.index_add_(0, lfs_indices, lfs_src);
    torch_ratios.index_add_(0, torch_indices, torch_src);

    // Compare results
    assertTensorsEqual(lfs_ratios, torch_ratios);

    // Also check that we actually incremented values
    auto lfs_cpu = lfs_ratios.cpu();
    const float* data = lfs_cpu.ptr<float>();

    // Count how many values are > 1 (should match number of unique indices)
    int count_incremented = 0;
    for (int i = 0; i < N; ++i) {
        if (data[i] > 1.0f) {
            count_incremented++;
        }
    }

    EXPECT_GT(count_incremented, 0) << "No values were incremented - index_add_ failed!";
}

TEST_F(MCMCCriticalBugs_IndexAdd, Int32IndicesBaseline) {
    const int N = 1000;
    const int num_samples = 100;

    auto lfs_ratios = Tensor::ones({N}, Device::CUDA, DataType::Float32);
    auto torch_ratios = torch::ones({N}, torch::kFloat32).cuda();

    std::vector<int32_t> idx_data(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        idx_data[i] = rand() % N;
    }

    auto lfs_indices = Tensor::from_blob(idx_data.data(), {num_samples}, Device::CPU, DataType::Int32).cuda();
    auto torch_indices = torch::from_blob(idx_data.data(), {num_samples}, torch::kInt32).cuda();

    auto lfs_src = Tensor::ones({num_samples}, Device::CUDA, DataType::Float32);
    auto torch_src = torch::ones({num_samples}, torch::kFloat32).cuda();

    lfs_ratios.index_add_(0, lfs_indices, lfs_src);
    torch_ratios.index_add_(0, torch_indices, torch_src);

    assertTensorsEqual(lfs_ratios, torch_ratios);
}

TEST_F(MCMCCriticalBugs_IndexAdd, MCMCRatioCountingScenario) {
    // Simulate the exact MCMC scenario that exposed the bug
    const int N = 10000;  // Total Gaussians
    const int n_dead = 1000;  // Dead Gaussians to relocate

    // Step 1: Create initial ratios (all ones)
    auto lfs_ratios = Tensor::ones({N}, Device::CUDA, DataType::Float32);
    auto torch_ratios = torch::ones({N}, torch::kFloat32).cuda();

    // Step 2: Sample indices (with replacement) - returns int64
    std::vector<int64_t> sampled_indices(n_dead);
    for (int i = 0; i < n_dead; ++i) {
        sampled_indices[i] = rand() % N;
    }

    auto lfs_sampled_idxs = Tensor::from_blob(sampled_indices.data(), {n_dead},
                                               Device::CPU, DataType::Int64).cuda();
    auto torch_sampled_idxs = torch::from_blob(sampled_indices.data(), {n_dead},
                                                torch::kInt64).cuda();

    // Step 3: Add ones at sampled indices (count occurrences)
    auto lfs_ones = Tensor::ones({n_dead}, Device::CUDA, DataType::Float32);
    auto torch_ones = torch::ones({n_dead}, torch::kFloat32).cuda();

    lfs_ratios.index_add_(0, lfs_sampled_idxs, lfs_ones);
    torch_ratios.index_add_(0, torch_sampled_idxs, torch_ones);

    // Step 4: Select ratios at sampled indices
    auto lfs_result = lfs_ratios.index_select(0, lfs_sampled_idxs);
    auto torch_result = torch_ratios.index_select(0, torch_sampled_idxs);

    // Compare float results
    assertTensorsEqual(lfs_result, torch_result);

    // CRITICAL CHECK: All values should be >= 2 (1 initial + at least 1 from index_add)
    auto lfs_cpu = lfs_result.cpu();
    const float* data = lfs_cpu.ptr<float>();

    float min_val = *std::min_element(data, data + n_dead);
    float max_val = *std::max_element(data, data + n_dead);

    EXPECT_GE(min_val, 2.0f) << "Bug detected: min ratio is " << min_val << ", should be >= 2";
    EXPECT_LE(max_val, 52.0f) << "Max ratio exceeded n_max+1 = 52";

    std::cout << "Ratio range: [" << min_val << ", " << max_val << "]" << std::endl;
}

TEST_F(MCMCCriticalBugs_IndexAdd, FloatTensorInt64Indices) {
    const int N = 500;
    const int num_samples = 50;

    auto lfs_tensor = Tensor::zeros({N}, Device::CUDA, DataType::Float32);
    auto torch_tensor = torch::zeros({N}, torch::kFloat32).cuda();

    std::vector<int64_t> idx_data(num_samples);
    std::vector<float> values(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        idx_data[i] = rand() % N;
        values[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    auto lfs_indices = Tensor::from_blob(idx_data.data(), {num_samples}, Device::CPU, DataType::Int64).cuda();
    auto torch_indices = torch::from_blob(idx_data.data(), {num_samples}, torch::kInt64).cuda();

    auto lfs_src = Tensor::from_blob(values.data(), {num_samples}, Device::CPU, DataType::Float32).cuda();
    auto torch_src = torch::from_blob(values.data(), {num_samples}, torch::kFloat32).cuda();

    lfs_tensor.index_add_(0, lfs_indices, lfs_src);
    torch_tensor.index_add_(0, torch_indices, torch_src);

    assertTensorsEqual(lfs_tensor, torch_tensor);
}
