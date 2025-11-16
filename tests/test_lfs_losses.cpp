/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "optimizer/adam_optimizer.hpp"  // For src/training_new include path
#include "losses/regularization.hpp"      // New libtorch-free losses
#include "losses/photometric_loss.hpp"    // New libtorch-free losses
#include "kernels/regularization.cuh"     // Old CUDA kernels for comparison
#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <torch/torch.h>

using namespace lfs::core;
using namespace lfs::training::losses;

namespace {

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

// Helper to convert torch::Tensor to lfs::core::Tensor
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

// Compare floats with tolerance
bool float_close(float a, float b, float rtol = 1e-5f, float atol = 1e-5f) {
    float diff = std::abs(a - b);
    float threshold = atol + rtol * std::abs(b);
    return diff <= threshold;
}

// Compare tensors with tolerance
bool tensors_close(const lfs::core::Tensor& lfs_tensor, const torch::Tensor& torch_tensor,
                   float rtol = 1e-5f, float atol = 1e-5f) {
    if (lfs_tensor.numel() != torch_tensor.numel()) return false;

    auto lfs_vec = lfs_tensor.cpu().to_vector();
    auto torch_cpu = torch_tensor.cpu().contiguous();
    auto torch_ptr = torch_cpu.data_ptr<float>();

    for (size_t i = 0; i < lfs_vec.size(); i++) {
        if (!float_close(lfs_vec[i], torch_ptr[i], rtol, atol)) {
            return false;
        }
    }
    return true;
}

// ===================================================================================
// ScaleRegularization Tests
// ===================================================================================

TEST(LfsLossesTest, ScaleRegularization_Basic) {
    const size_t n = 1000;
    const float weight = 0.01f;

    // Create random scaling_raw
    auto scaling_raw = Tensor::randn({n, 3}, Device::CUDA);
    auto scaling_raw_grad = Tensor::zeros({n, 3}, Device::CUDA);

    // Compute with new implementation
    ScaleRegularization::Params params{.weight = weight};
    auto result = ScaleRegularization::forward(scaling_raw, scaling_raw_grad, params);

    ASSERT_TRUE(result.has_value()) << result.error();
    float new_loss = result->item<float>();  // Sync tensor to CPU

    // Compute with old implementation
    auto torch_scaling_raw = to_torch(scaling_raw);
    torch_scaling_raw.set_requires_grad(true);

    auto old_loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
        torch_scaling_raw, weight);

    // Compare loss values
    EXPECT_TRUE(float_close(new_loss, old_loss, 1e-5f, 1e-6f))
        << "New loss: " << new_loss << " vs Old loss: " << old_loss;

    // Compare gradients
    EXPECT_TRUE(tensors_close(scaling_raw_grad, torch_scaling_raw.grad(), 1e-5f, 1e-6f))
        << "Gradients don't match!";
}

TEST(LfsLossesTest, ScaleRegularization_ZeroWeight) {
    auto scaling_raw = Tensor::randn({100, 3}, Device::CUDA);
    auto scaling_raw_grad = Tensor::zeros({100, 3}, Device::CUDA);

    ScaleRegularization::Params params{.weight = 0.0f};
    auto result = ScaleRegularization::forward(scaling_raw, scaling_raw_grad, params);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->item<float>(), 0.0f);

    // Gradient should be unchanged (all zeros)
    auto grad_vec = scaling_raw_grad.cpu().to_vector();
    for (float val : grad_vec) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }
}

TEST(LfsLossesTest, ScaleRegularization_LargeScale) {
    const size_t n = 100000;
    auto scaling_raw = Tensor::randn({n, 3}, Device::CUDA);
    auto scaling_raw_grad = Tensor::zeros({n, 3}, Device::CUDA);

    ScaleRegularization::Params params{.weight = 0.1f};
    auto result = ScaleRegularization::forward(scaling_raw, scaling_raw_grad, params);

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->item<float>(), 0.0f);

    // Verify gradient is non-zero
    auto grad_sum = scaling_raw_grad.abs().sum();
    EXPECT_GT(grad_sum.item(), 0.0f);
}

// ===================================================================================
// OpacityRegularization Tests
// ===================================================================================

TEST(LfsLossesTest, OpacityRegularization_Basic) {
    const size_t n = 1000;
    const float weight = 0.01f;

    auto opacity_raw = Tensor::randn({n, 1}, Device::CUDA);
    auto opacity_raw_grad = Tensor::zeros({n, 1}, Device::CUDA);

    // New implementation
    OpacityRegularization::Params params{.weight = weight};
    auto result = OpacityRegularization::forward(opacity_raw, opacity_raw_grad, params);

    ASSERT_TRUE(result.has_value()) << result.error();
    float new_loss = result->item<float>();  // Sync tensor to CPU

    // Old implementation
    auto torch_opacity_raw = to_torch(opacity_raw);
    torch_opacity_raw.set_requires_grad(true);

    float old_loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
        torch_opacity_raw, weight);

    // Compare
    EXPECT_TRUE(float_close(new_loss, old_loss, 1e-5f, 1e-6f))
        << "New: " << new_loss << " vs Old: " << old_loss;
    EXPECT_TRUE(tensors_close(opacity_raw_grad, torch_opacity_raw.grad(), 1e-5f, 1e-6f));
}

TEST(LfsLossesTest, OpacityRegularization_ZeroWeight) {
    auto opacity_raw = Tensor::randn({100, 1}, Device::CUDA);
    auto opacity_raw_grad = Tensor::zeros({100, 1}, Device::CUDA);

    OpacityRegularization::Params params{.weight = 0.0f};
    auto result = OpacityRegularization::forward(opacity_raw, opacity_raw_grad, params);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->item<float>(), 0.0f);
}

TEST(LfsLossesTest, OpacityRegularization_GradientAccumulation) {
    auto opacity_raw = Tensor::randn({100, 1}, Device::CUDA);
    auto opacity_raw_grad = Tensor::ones({100, 1}, Device::CUDA);  // Pre-existing gradient

    auto grad_before = opacity_raw_grad.clone();

    OpacityRegularization::Params params{.weight = 0.01f};
    auto result = OpacityRegularization::forward(opacity_raw, opacity_raw_grad, params);

    ASSERT_TRUE(result.has_value());

    // Gradient should have been accumulated (not replaced)
    auto diff = (opacity_raw_grad - grad_before).abs().sum();
    EXPECT_GT(diff.item(), 0.0f) << "Gradient should have changed";
}

// ===================================================================================
// PhotometricLoss Tests
// ===================================================================================

TEST(LfsLossesTest, PhotometricLoss_PureL1) {
    const int H = 64, W = 64, C = 3;

    auto rendered = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);
    auto gt_image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);

    // Pure L1 (lambda_dssim = 0.0)
    PhotometricLoss::Params params{.lambda_dssim = 0.0f};
    PhotometricLoss loss_fn;
    auto result = loss_fn.forward(rendered, gt_image, params);

    ASSERT_TRUE(result.has_value()) << result.error();
    auto [loss_tensor, ctx] = *result;

    // Manually compute L1 loss
    auto diff = (rendered - gt_image).abs();
    float expected_loss = diff.mean().item();

    // Sync loss to CPU for comparison
    float loss = loss_tensor.item();

    EXPECT_TRUE(float_close(loss, expected_loss, 1e-5f, 1e-6f))
        << "L1 loss: " << loss << " vs expected: " << expected_loss;

    // Gradient should be sign(diff) / N
    auto expected_grad = (rendered - gt_image).sign() / static_cast<float>(rendered.numel());
    EXPECT_TRUE(tensors_close(ctx.grad_image, to_torch(expected_grad), 1e-5f, 1e-6f));
}

TEST(LfsLossesTest, PhotometricLoss_PureSSIM) {
    const int H = 128, W = 128, C = 3;

    auto rendered = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);
    auto gt_image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);

    // Pure SSIM (lambda_dssim = 1.0)
    PhotometricLoss::Params params{.lambda_dssim = 1.0f};
    PhotometricLoss loss_fn;
    auto result = loss_fn.forward(rendered, gt_image, params);

    ASSERT_TRUE(result.has_value()) << result.error();
    auto [loss_tensor, ctx] = *result;

    // Sync to CPU for comparison
    float loss = loss_tensor.item();

    // Loss should be in range [0, 2] (1 - SSIM, where SSIM is in [-1, 1])
    EXPECT_GE(loss, 0.0f);
    EXPECT_LE(loss, 2.0f);

    // Gradient should be non-zero
    auto grad_sum_tensor = ctx.grad_image.abs().sum();
    auto grad_vec = grad_sum_tensor.cpu().to_vector();
    float grad_norm = grad_vec[0];  // sum() returns a single element tensor
    EXPECT_GT(grad_norm, 0.0f);
}

TEST(LfsLossesTest, PhotometricLoss_Combined) {
    const int H = 64, W = 64, C = 3;

    auto rendered = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);
    auto gt_image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);

    // Combined loss (lambda_dssim = 0.2, typical value)
    PhotometricLoss::Params params{.lambda_dssim = 0.2f};
    PhotometricLoss loss_fn;
    auto result = loss_fn.forward(rendered, gt_image, params);

    ASSERT_TRUE(result.has_value()) << result.error();
    auto [combined_loss_tensor, combined_ctx] = *result;

    // IMPORTANT: Synchronize to ensure all GPU operations are complete
    cudaDeviceSynchronize();

    // IMPORTANT: Read combined loss BEFORE calling forward() again (which reuses buffers)
    float combined_loss = combined_loss_tensor.item();

    // Compute pure L1
    params.lambda_dssim = 0.0f;
    auto l1_result = loss_fn.forward(rendered, gt_image, params);
    ASSERT_TRUE(l1_result.has_value());
    float l1_loss = l1_result->first.item();

    // Compute pure SSIM
    params.lambda_dssim = 1.0f;
    auto ssim_result = loss_fn.forward(rendered, gt_image, params);
    ASSERT_TRUE(ssim_result.has_value());
    float ssim_loss = ssim_result->first.item();

    // Combined should be weighted average
    float expected_combined = 0.8f * l1_loss + 0.2f * ssim_loss;

    EXPECT_TRUE(float_close(combined_loss, expected_combined, 1e-4f, 1e-5f))
        << "Combined: " << combined_loss << " vs expected: " << expected_combined;
}

TEST(LfsLossesTest, PhotometricLoss_IdenticalImages) {
    const int H = 32, W = 32, C = 3;

    auto image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);

    PhotometricLoss::Params params{.lambda_dssim = 0.2f};
    PhotometricLoss loss_fn;
    auto result = loss_fn.forward(image, image, params);

    ASSERT_TRUE(result.has_value());
    auto [loss_tensor, ctx] = *result;

    // Sync to CPU for comparison
    float loss = loss_tensor.item();

    // Loss should be very close to zero for identical images
    EXPECT_LT(loss, 1e-4f) << "Loss for identical images should be near zero";
}

TEST(LfsLossesTest, PhotometricLoss_ShapeMismatch) {
    auto rendered = Tensor::rand({64, 64, 3}, Device::CUDA);
    auto gt_image = Tensor::rand({32, 32, 3}, Device::CUDA);

    PhotometricLoss::Params params{.lambda_dssim = 0.2f};
    PhotometricLoss loss_fn;
    auto result = loss_fn.forward(rendered, gt_image, params);

    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Shape mismatch") != std::string::npos);
}

// ===================================================================================
// Gradient Consistency Tests
// ===================================================================================

TEST(LfsLossesTest, GradientConsistency_ScaleReg) {
    const int n_trials = 5;
    const size_t n = 500;
    const float weight = 0.05f;

    for (int trial = 0; trial < n_trials; trial++) {
        auto scaling_raw = Tensor::randn({n, 3}, Device::CUDA);
        auto scaling_raw_grad_new = Tensor::zeros({n, 3}, Device::CUDA);

        // New implementation
        ScaleRegularization::Params params{.weight = weight};
        auto result = ScaleRegularization::forward(scaling_raw, scaling_raw_grad_new, params);
        ASSERT_TRUE(result.has_value());

        // Old implementation
        auto torch_scaling_raw = to_torch(scaling_raw);
        torch_scaling_raw.set_requires_grad(true);
        gs::regularization::compute_exp_l1_regularization_with_grad_cuda(torch_scaling_raw, weight);

        // Gradients should match exactly (same kernel)
        EXPECT_TRUE(tensors_close(scaling_raw_grad_new, torch_scaling_raw.grad(), 1e-6f, 1e-7f))
            << "Trial " << trial << " failed";
    }
}

TEST(LfsLossesTest, GradientConsistency_OpacityReg) {
    const int n_trials = 5;
    const size_t n = 500;
    const float weight = 0.05f;

    for (int trial = 0; trial < n_trials; trial++) {
        auto opacity_raw = Tensor::randn({n, 1}, Device::CUDA);
        auto opacity_raw_grad_new = Tensor::zeros({n, 1}, Device::CUDA);

        // New implementation
        OpacityRegularization::Params params{.weight = weight};
        auto result = OpacityRegularization::forward(opacity_raw, opacity_raw_grad_new, params);
        ASSERT_TRUE(result.has_value());

        // Old implementation
        auto torch_opacity_raw = to_torch(opacity_raw);
        torch_opacity_raw.set_requires_grad(true);
        gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(torch_opacity_raw, weight);

        EXPECT_TRUE(tensors_close(opacity_raw_grad_new, torch_opacity_raw.grad(), 1e-6f, 1e-7f))
            << "Trial " << trial << " failed";
    }
}

// ===================================================================================
// Edge Cases
// ===================================================================================

TEST(LfsLossesTest, EdgeCase_SingleElement) {
    // Scale regularization with single element
    auto scaling_raw = Tensor::randn({1, 3}, Device::CUDA);
    auto scaling_raw_grad = Tensor::zeros({1, 3}, Device::CUDA);

    ScaleRegularization::Params params{.weight = 0.1f};
    auto result = ScaleRegularization::forward(scaling_raw, scaling_raw_grad, params);

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->item<float>(), 0.0f);
}

TEST(LfsLossesTest, EdgeCase_VeryLargeWeight) {
    auto scaling_raw = Tensor::randn({100, 3}, Device::CUDA);
    auto scaling_raw_grad = Tensor::zeros({100, 3}, Device::CUDA);

    ScaleRegularization::Params params{.weight = 1000.0f};
    auto result = ScaleRegularization::forward(scaling_raw, scaling_raw_grad, params);

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->item<float>(), 0.0f);
}

TEST(LfsLossesTest, EdgeCase_SmallImage) {
    // Minimum size image for SSIM (needs some padding)
    const int H = 16, W = 16, C = 3;

    auto rendered = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);
    auto gt_image = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);

    PhotometricLoss::Params params{.lambda_dssim = 0.2f};
    PhotometricLoss loss_fn;
    auto result = loss_fn.forward(rendered, gt_image, params);

    ASSERT_TRUE(result.has_value());
}

} // anonymous namespace
