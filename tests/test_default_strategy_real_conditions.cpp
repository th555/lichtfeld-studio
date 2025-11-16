/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// LFS (LibTorch-free) implementation
#include "strategies/default_strategy.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include "core_new/parameters.hpp"

// Reference (LibTorch) implementation
#include "training/strategies/default_strategy.hpp"  // gs::training::DefaultStrategy
#include "training/rasterization/rasterizer.hpp"  // gs::training::RenderOutput
#include "core/splat_data.hpp"
#include "core/parameters.hpp"

using namespace lfs::core;
using namespace lfs::training;

namespace {

// ===================================================================================
// Helper functions for Torch-Tensor interop (reused from test_mcmc_strategy.cpp)
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

// Compare tensors with tolerance
bool tensors_close(const lfs::core::Tensor& lfs_tensor, const torch::Tensor& torch_tensor,
                   float rtol = 1e-4f, float atol = 1e-5f) {
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

float mean_diff(const lfs::core::Tensor& lfs_tensor, const torch::Tensor& torch_tensor) {
    auto lfs_vec = lfs_tensor.cpu().to_vector();
    auto torch_cpu = torch_tensor.cpu().contiguous();
    auto torch_ptr = torch_cpu.data_ptr<float>();

    float sum = 0.0f;
    for (size_t i = 0; i < lfs_vec.size(); i++) {
        sum += std::abs(lfs_vec[i] - torch_ptr[i]);
    }
    return sum / lfs_vec.size();
}

// Helper to initialize optimizer state by running a few steps
void warm_up_optimizers(lfs::training::DefaultStrategy& lfs_default, gs::training::DefaultStrategy& gs_default, int num_steps = 5) {
    for (int i = 0; i < num_steps; i++) {
        // Set fake gradients (same for both)
        lfs_default.get_model().means_grad().fill_(0.001f);
        lfs_default.get_model().sh0_grad().fill_(0.001f);
        lfs_default.get_model().shN_grad().fill_(0.0005f);
        lfs_default.get_model().scaling_grad().fill_(0.001f);
        lfs_default.get_model().rotation_grad().fill_(0.001f);
        lfs_default.get_model().opacity_grad().fill_(0.001f);

        // For reference impl, allocate gradients using mutable_grad()
        auto& gs_model = gs_default.get_model();
        if (!gs_model.means().grad().defined()) {
            gs_model.means().mutable_grad() = torch::zeros_like(gs_model.means());
        }
        if (!gs_model.sh0().grad().defined()) {
            gs_model.sh0().mutable_grad() = torch::zeros_like(gs_model.sh0());
        }
        if (!gs_model.shN().grad().defined()) {
            gs_model.shN().mutable_grad() = torch::zeros_like(gs_model.shN());
        }
        if (!gs_model.scaling_raw().grad().defined()) {
            gs_model.scaling_raw().mutable_grad() = torch::zeros_like(gs_model.scaling_raw());
        }
        if (!gs_model.rotation_raw().grad().defined()) {
            gs_model.rotation_raw().mutable_grad() = torch::zeros_like(gs_model.rotation_raw());
        }
        if (!gs_model.opacity_raw().grad().defined()) {
            gs_model.opacity_raw().mutable_grad() = torch::zeros_like(gs_model.opacity_raw());
        }

        gs_model.means().mutable_grad().fill_(0.001f);
        gs_model.sh0().mutable_grad().fill_(0.001f);
        gs_model.shN().mutable_grad().fill_(0.0005f);
        gs_model.scaling_raw().mutable_grad().fill_(0.001f);
        gs_model.rotation_raw().mutable_grad().fill_(0.001f);
        gs_model.opacity_raw().mutable_grad().fill_(0.001f);

        // Step optimizer
        lfs_default.step(i);
        gs_default.step(i);
    }
}

// Helper to create test parameters
lfs::core::param::OptimizationParameters create_test_params() {
    lfs::core::param::OptimizationParameters params;
    params.iterations = 30000;
    params.means_lr = 1.6e-4f;
    params.min_opacity = 0.005f;
    params.max_cap = 1000000;
    params.start_refine = 500;
    params.stop_refine = 15000;
    params.refine_every = 100;
    params.sh_degree_interval = 1000;
    params.grad_threshold = 0.0002f;  // Default strategy specific
    return params;
}

gs::param::OptimizationParameters to_gs_params(const lfs::core::param::OptimizationParameters& lfs_params) {
    gs::param::OptimizationParameters gs_params;
    gs_params.iterations = lfs_params.iterations;
    gs_params.means_lr = lfs_params.means_lr;
    gs_params.min_opacity = lfs_params.min_opacity;
    gs_params.max_cap = lfs_params.max_cap;
    gs_params.start_refine = lfs_params.start_refine;
    gs_params.stop_refine = lfs_params.stop_refine;
    gs_params.refine_every = lfs_params.refine_every;
    gs_params.sh_degree_interval = lfs_params.sh_degree_interval;
    gs_params.grad_threshold = lfs_params.grad_threshold;
    return gs_params;
}

// Helper function to create a simple SplatData for testing
lfs::core::SplatData create_lfs_splat_data(size_t n_points = 100, int sh_degree = 3) {
    auto means = Tensor::randn({n_points, 3}, Device::CUDA);
    auto sh0 = Tensor::randn({n_points, 1, 3}, Device::CUDA);

    // For sh_degree=3, we have (3+1)^2 = 16 coefficients, minus DC = 15
    size_t sh_rest_coeffs = (sh_degree + 1) * (sh_degree + 1) - 1;
    auto shN = Tensor::randn({n_points, sh_rest_coeffs, 3}, Device::CUDA);

    auto scaling = Tensor::randn({n_points, 3}, Device::CUDA);
    auto rotation = Tensor::randn({n_points, 4}, Device::CUDA);

    // Normalize rotation quaternions
    auto rot_norm = (rotation * rotation).sum(-1, true).sqrt();
    rotation = rotation / rot_norm;

    auto opacity = Tensor::randn({n_points, 1}, Device::CUDA);

    return lfs::core::SplatData(sh_degree, std::move(means), std::move(sh0), std::move(shN),
                                 std::move(scaling), std::move(rotation), std::move(opacity), 1.0f);
}

// Create reference gs::SplatData from lfs::core::SplatData (same initial state)
gs::SplatData create_gs_splat_data(const lfs::core::SplatData& lfs_splat) {
    auto means = to_torch(lfs_splat.means()).set_requires_grad(true);
    auto sh0 = to_torch(lfs_splat.sh0()).set_requires_grad(true);
    auto shN = to_torch(lfs_splat.shN()).set_requires_grad(true);
    auto scaling = to_torch(lfs_splat.scaling_raw()).set_requires_grad(true);
    auto rotation = to_torch(lfs_splat.rotation_raw()).set_requires_grad(true);
    auto opacity = to_torch(lfs_splat.opacity_raw()).set_requires_grad(true);

    // Use max_sh_degree, not active (active starts at 0)
    return gs::SplatData(lfs_splat.get_max_sh_degree(), means, sh0, shN,
                        scaling, rotation, opacity, 1.0f);  // percent_dense = 1.0
}

// ===================================================================================
// Test Suite: Default Strategy Correctness
// ===================================================================================

TEST(DefaultStrategyTest, Initialization) {
    // Test that default strategy initializes correctly
    auto lfs_splat = create_lfs_splat_data(100, 3);
    lfs_splat.allocate_gradients();

    lfs::training::DefaultStrategy lfs_default(std::move(lfs_splat));

    auto params = create_test_params();
    lfs_default.initialize(params);

    // Check that model is accessible
    EXPECT_EQ(lfs_default.get_model().size(), 100);
    // SH degree starts at 0 after initialization
    EXPECT_EQ(lfs_default.get_model().get_active_sh_degree(), 0);
}

TEST(DefaultStrategyTest, IsRefining) {
    auto lfs_splat = create_lfs_splat_data(100, 3);
    lfs_splat.allocate_gradients();

    lfs::training::DefaultStrategy lfs_default(std::move(lfs_splat));

    auto params = create_test_params();
    params.start_refine = 500;
    params.stop_refine = 15000;
    params.refine_every = 100;
    lfs_default.initialize(params);

    // Before start_refine
    EXPECT_FALSE(lfs_default.is_refining(0));
    EXPECT_FALSE(lfs_default.is_refining(499));

    // During refining window (on boundary)
    EXPECT_TRUE(lfs_default.is_refining(500));
    EXPECT_TRUE(lfs_default.is_refining(600));
    EXPECT_FALSE(lfs_default.is_refining(650));  // Not on boundary
    EXPECT_TRUE(lfs_default.is_refining(1000));
    EXPECT_TRUE(lfs_default.is_refining(14900));

    // After stop_refine
    EXPECT_FALSE(lfs_default.is_refining(15000));
    EXPECT_FALSE(lfs_default.is_refining(20000));
}

TEST(DefaultStrategyTest, DuplicateGaussians_WithOptimizerState) {
    // Test duplicate operation with warm optimizer state
    std::cout << "Creating splat data..." << std::endl;
    auto lfs_splat = create_lfs_splat_data(500, 3);
    lfs_splat.allocate_gradients();

    auto gs_splat = create_gs_splat_data(lfs_splat);

    lfs::training::DefaultStrategy lfs_default(std::move(lfs_splat));
    gs::training::DefaultStrategy gs_default(std::move(gs_splat));

    auto lfs_params = create_test_params();
    lfs_params.grad_threshold = 0.0001f;  // Low threshold to trigger duplication
    auto gs_params = to_gs_params(lfs_params);

    lfs_default.initialize(lfs_params);
    gs_default.initialize(gs_params);

    // Warm up optimizers with fake gradients
    std::cout << "Warming up optimizers..." << std::endl;
    warm_up_optimizers(lfs_default, gs_default, 5);

    // Set high gradients to trigger duplication
    auto& lfs_model = lfs_default.get_model();
    auto& gs_model = gs_default.get_model();

    lfs_model.means_grad().fill_(0.01f);  // High gradients
    gs_model.means().mutable_grad().fill_(0.01f);

    // Trigger densification via post_backward
    lfs::training::RenderOutput lfs_render_out;
    gs::training::RenderOutput gs_render_out;

    int iter = 600;  // Refinement iteration
    ASSERT_TRUE(lfs_default.is_refining(iter));

    std::cout << "Calling post_backward at iter " << iter << std::endl;
    lfs_default.post_backward(iter, lfs_render_out);
    gs_default.post_backward(iter, gs_render_out);

    std::cout << "LFS model size after densification: " << lfs_model.size() << std::endl;
    std::cout << "GS model size after densification: " << gs_model.size() << std::endl;

    // Validate results
    EXPECT_GT(lfs_model.size(), 500) << "LFS model should have grown through duplication";
    EXPECT_GT(gs_model.size(), 500) << "GS model should have grown through duplication";

    // Check no NaN/inf values exist in any parameters
    auto check_valid = [](const lfs::core::Tensor& t, const std::string& name) {
        auto vec = t.cpu().to_vector();
        EXPECT_GT(vec.size(), 0) << name << " should not be empty";
        for (size_t i = 0; i < vec.size(); i++) {
            EXPECT_FALSE(std::isnan(vec[i])) << name << "[" << i << "] is NaN";
            EXPECT_FALSE(std::isinf(vec[i])) << name << "[" << i << "] is inf";
        }
    };

    check_valid(lfs_model.means(), "LFS means");
    check_valid(lfs_model.get_opacity(), "LFS opacity");
    check_valid(lfs_model.get_scaling(), "LFS scaling");

    std::cout << "Duplicate test passed - model grew from 500 to " << lfs_model.size() << " Gaussians" << std::endl;
}

TEST(DefaultStrategyTest, SplitGaussians_WithOptimizerState) {
    // Test split operation with warm optimizer state
    std::cout << "Creating splat data..." << std::endl;
    auto lfs_splat = create_lfs_splat_data(500, 3);
    lfs_splat.allocate_gradients();

    auto gs_splat = create_gs_splat_data(lfs_splat);

    lfs::training::DefaultStrategy lfs_default(std::move(lfs_splat));
    gs::training::DefaultStrategy gs_default(std::move(gs_splat));

    auto lfs_params = create_test_params();
    lfs_params.grad_threshold = 0.0001f;  // Low threshold
    auto gs_params = to_gs_params(lfs_params);

    lfs_default.initialize(lfs_params);
    gs_default.initialize(gs_params);

    // Warm up optimizers
    std::cout << "Warming up optimizers..." << std::endl;
    warm_up_optimizers(lfs_default, gs_default, 5);

    // Set high gradients and large scales to trigger split
    auto& lfs_model = lfs_default.get_model();
    auto& gs_model = gs_default.get_model();

    lfs_model.means_grad().fill_(0.01f);  // High gradients
    lfs_model.scaling_raw().fill_(10.0f);  // Large scales
    gs_model.means().mutable_grad().fill_(0.01f);
    gs_model.scaling_raw().fill_(10.0f);

    // Trigger densification
    lfs::training::RenderOutput lfs_render_out;
    gs::training::RenderOutput gs_render_out;

    int iter = 600;
    ASSERT_TRUE(lfs_default.is_refining(iter));

    std::cout << "Calling post_backward at iter " << iter << std::endl;
    lfs_default.post_backward(iter, lfs_render_out);
    gs_default.post_backward(iter, gs_render_out);

    std::cout << "LFS model size after split: " << lfs_model.size() << std::endl;
    std::cout << "GS model size after split: " << gs_model.size() << std::endl;

    // Validate
    EXPECT_GT(lfs_model.size(), 500) << "LFS model should have grown through split";
    EXPECT_GT(gs_model.size(), 500) << "GS model should have grown through split";

    auto check_valid = [](const lfs::core::Tensor& t, const std::string& name) {
        auto vec = t.cpu().to_vector();
        EXPECT_GT(vec.size(), 0) << name << " should not be empty";
        for (size_t i = 0; i < vec.size(); i++) {
            EXPECT_FALSE(std::isnan(vec[i])) << name << "[" << i << "] is NaN";
            EXPECT_FALSE(std::isinf(vec[i])) << name << "[" << i << "] is inf";
        }
    };

    check_valid(lfs_model.means(), "LFS means after split");
    check_valid(lfs_model.sh0(), "LFS sh0 after split");
    check_valid(lfs_model.shN(), "LFS shN after split");

    std::cout << "Split test passed - model grew from 500 to " << lfs_model.size() << " Gaussians" << std::endl;
}

TEST(DefaultStrategyStressTest, FullTrainingLoop_100Iterations) {
    // Simulate 100 iterations of training with densification
    std::cout << "Creating splat data for stress test..." << std::endl;
    auto lfs_splat = create_lfs_splat_data(200, 3);
    lfs_splat.allocate_gradients();

    lfs::training::DefaultStrategy lfs_default(std::move(lfs_splat));

    auto params = create_test_params();
    params.start_refine = 50;
    params.stop_refine = 5000;
    params.refine_every = 10;
    params.sh_degree_interval = 25;
    params.max_cap = 500;
    params.grad_threshold = 0.0001f;  // Low to trigger operations

    lfs_default.initialize(params);

    auto& model = lfs_default.get_model();
    lfs::training::RenderOutput render_out;

    // Track metrics throughout training
    std::vector<int> sizes;
    std::vector<int> sh_degrees;

    std::cout << "Running 100 iteration training loop..." << std::endl;
    for (int iter = 0; iter < 100; iter++) {
        // Simulate gradients
        model.means_grad().fill_(0.001f);
        model.sh0_grad().fill_(0.001f);
        model.shN_grad().fill_(0.0005f);
        model.scaling_grad().fill_(0.001f);
        model.rotation_grad().fill_(0.001f);
        model.opacity_grad().fill_(0.001f);

        // Step optimizer
        lfs_default.step(iter);

        // Post-backward (refinement happens here)
        lfs_default.post_backward(iter, render_out);

        sizes.push_back(model.size());
        sh_degrees.push_back(model.get_active_sh_degree());

        // Validate no corruption every 10 iterations
        if (iter % 10 == 0) {
            auto means = model.means().cpu().to_vector();
            auto opacity = model.get_opacity().cpu().to_vector();
            auto scaling = model.get_scaling().cpu().to_vector();

            EXPECT_GT(means.size(), 0) << "Means empty at iter " << iter;
            EXPECT_GT(opacity.size(), 0) << "Opacity empty at iter " << iter;
            EXPECT_GT(scaling.size(), 0) << "Scaling empty at iter " << iter;

            // Check for NaN/Inf
            for (size_t i = 0; i < std::min(opacity.size(), size_t(10)); i++) {
                EXPECT_FALSE(std::isnan(opacity[i])) << "NaN in opacity at iter " << iter;
                EXPECT_FALSE(std::isinf(opacity[i])) << "Inf in opacity at iter " << iter;
            }

            std::cout << "Iter " << iter << ": size=" << model.size()
                      << ", sh_degree=" << model.get_active_sh_degree() << std::endl;
        }
    }

    // Final validation
    EXPECT_GT(model.size(), 200) << "Model should have grown";
    EXPECT_LE(model.size(), 500) << "Model should respect max_cap";
    EXPECT_GT(model.get_active_sh_degree(), 0) << "SH degree should have incremented";
    EXPECT_LE(model.get_active_sh_degree(), 3) << "SH degree should not exceed max";

    std::cout << "Long training test passed: " << sizes.front() << " -> " << sizes.back()
              << " Gaussians, SH degree " << sh_degrees.front() << " -> " << sh_degrees.back() << std::endl;
}

} // namespace
