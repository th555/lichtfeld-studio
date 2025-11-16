/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

// LibTorch-free implementation
#include "training_new/components/sparsity_optimizer.hpp"
#include "core_new/tensor.hpp"

// Reference LibTorch implementation
#include "training/components/sparsity_optimizer.hpp"

namespace {

// Helper to clear any pending CUDA errors (workaround for LFS implementation not checking CUDA errors)
inline void clear_cuda_errors() {
    cudaDeviceSynchronize();
    cudaGetLastError(); // Clears the error
}

// Helper to convert lfs::core::Tensor to torch::Tensor
torch::Tensor to_torch(const lfs::core::Tensor& lfs_tensor) {
    // Ensure CUDA operations are complete before reading
    if (lfs_tensor.device() == lfs::core::Device::CUDA) {
        cudaDeviceSynchronize();
    }

    auto vec = lfs_tensor.cpu().to_vector();
    std::vector<int64_t> torch_shape;
    for (size_t i = 0; i < lfs_tensor.ndim(); i++) {
        torch_shape.push_back(lfs_tensor.shape()[i]);
    }

    // Create CPU tensor first
    auto cpu_tensor = torch::from_blob(vec.data(), torch_shape,
                                       torch::TensorOptions().dtype(torch::kFloat32)).clone();

    // Verify the clone succeeded
    if (!cpu_tensor.defined() || cpu_tensor.numel() == 0) {
        throw std::runtime_error("Failed to create torch tensor from LFS tensor");
    }

    // Move to CUDA if needed, with explicit contiguous() and clone() to ensure independence
    if (lfs_tensor.device() == lfs::core::Device::CUDA) {
        auto cuda_tensor = cpu_tensor.to(torch::kCUDA).contiguous().clone();
        // Ensure tensor is valid before returning
        if (!cuda_tensor.defined() || !cuda_tensor.is_cuda()) {
            throw std::runtime_error("Failed to move tensor to CUDA");
        }
        return cuda_tensor;
    }

    return cpu_tensor.contiguous().clone();
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
bool float_close(float a, float b, float rtol = 1e-4f, float atol = 1e-5f) {
    float diff = std::abs(a - b);
    float threshold = atol + rtol * std::abs(b);
    return diff <= threshold;
}

// Compare tensors with tolerance
bool tensors_close(const lfs::core::Tensor& lfs_tensor, const torch::Tensor& torch_tensor,
                   float rtol = 1e-4f, float atol = 1e-5f) {
    if (lfs_tensor.numel() != torch_tensor.numel()) {
        std::cerr << "Size mismatch: " << lfs_tensor.numel() << " vs " << torch_tensor.numel() << std::endl;
        return false;
    }

    auto lfs_vec = lfs_tensor.cpu().to_vector();
    auto torch_cpu = torch_tensor.cpu().contiguous();
    auto torch_ptr = torch_cpu.data_ptr<float>();

    size_t mismatch_count = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < lfs_vec.size(); i++) {
        float diff = std::abs(lfs_vec[i] - torch_ptr[i]);
        max_diff = std::max(max_diff, diff);
        if (!float_close(lfs_vec[i], torch_ptr[i], rtol, atol)) {
            mismatch_count++;
            if (mismatch_count <= 5) {  // Print first 5 mismatches
                std::cerr << "Mismatch at idx " << i << ": " << lfs_vec[i]
                         << " vs " << torch_ptr[i] << " (diff=" << diff << ")" << std::endl;
            }
        }
    }

    if (mismatch_count > 0) {
        std::cerr << "Total mismatches: " << mismatch_count << " / " << lfs_vec.size()
                  << " (max_diff=" << max_diff << ")" << std::endl;
    }
    return mismatch_count == 0;
}

// Copy internal state from reference to new implementation for fair comparison
void copy_state(lfs::training::ADMMSparsityOptimizer& new_opt, gs::training::ADMMSparsityOptimizer& ref_opt,
                const lfs::core::Tensor& opacities) {
    // Initialize reference first
    auto ref_opacities = to_torch(opacities);
    ref_opt.initialize(ref_opacities);

    // Initialize new optimizer
    new_opt.initialize(opacities);

    // Note: We can't directly copy internal state (u_, z_) as they're private
    // So we rely on initialization producing same results from same input
}

// ===================================================================================
// Basic Functionality Tests
// ===================================================================================

class SparsityOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available, skipping test";
        }
    }
};

TEST_F(SparsityOptimizerTest, Construction) {
    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .sparsify_steps = 1000,
        .init_rho = 0.001f,
        .prune_ratio = 0.5f,
        .update_every = 10,
        .start_iteration = 100
    };

    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .sparsify_steps = 1000,
        .init_rho = 0.001f,
        .prune_ratio = 0.5f,
        .update_every = 10,
        .start_iteration = 100
    };

    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    EXPECT_FALSE(new_opt.is_initialized());
    EXPECT_FALSE(ref_opt.is_initialized());

    EXPECT_FALSE(new_opt.should_update(50));
    EXPECT_FALSE(new_opt.should_apply_loss(50));
    EXPECT_FALSE(new_opt.should_prune(50));

    EXPECT_TRUE(new_opt.should_update(110));
    EXPECT_TRUE(new_opt.should_apply_loss(500));
    EXPECT_TRUE(new_opt.should_prune(1100));
}

TEST_F(SparsityOptimizerTest, Initialization) {
    const int n = 10000;

    lfs::training::ADMMSparsityOptimizer::Config new_config;
    gs::training::ADMMSparsityOptimizer::Config ref_config;

    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    // Create test opacities
    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    auto ref_opacities = to_torch(opacities);

    // Initialize both
    auto new_result = new_opt.initialize(opacities);
    auto ref_result = ref_opt.initialize(ref_opacities);

    ASSERT_TRUE(new_result.has_value()) << new_result.error();
    ASSERT_TRUE(ref_result.has_value()) << ref_result.error();

    EXPECT_TRUE(new_opt.is_initialized());
    EXPECT_TRUE(ref_opt.is_initialized());
}

// ===================================================================================
// Forward Pass Correctness Tests
// ===================================================================================

TEST_F(SparsityOptimizerTest, ForwardPass_SmallTensor) {
    const int n = 1000;

    lfs::training::ADMMSparsityOptimizer::Config new_config;
    gs::training::ADMMSparsityOptimizer::Config ref_config;
    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    // Create and initialize
    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    // Compute loss forward
    auto new_result = new_opt.compute_loss_forward(opacities);
    auto ref_result = ref_opt.compute_loss_forward(to_torch(opacities));

    ASSERT_TRUE(new_result.has_value()) << new_result.error();
    ASSERT_TRUE(ref_result.has_value()) << ref_result.error();

    auto [new_loss, new_ctx] = *new_result;
    auto [ref_loss, ref_ctx] = *ref_result;

    // Compare loss values
    EXPECT_TRUE(float_close(new_loss, ref_loss.template item<float>(), 1e-4f, 1e-5f))
        << "Loss mismatch: new=" << new_loss << " vs ref=" << ref_loss;
}

TEST_F(SparsityOptimizerTest, ForwardPass_MediumTensor) {
    const int n = 50000;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };
    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    auto new_result = new_opt.compute_loss_forward(opacities);
    auto ref_result = ref_opt.compute_loss_forward(to_torch(opacities));

    ASSERT_TRUE(new_result.has_value());
    ASSERT_TRUE(ref_result.has_value());

    auto [new_loss, new_ctx] = *new_result;
    auto [ref_loss, ref_ctx] = *ref_result;

    EXPECT_TRUE(float_close(new_loss, ref_loss.template item<float>(), 1e-4f, 1e-5f))
        << "Loss mismatch: new=" << new_loss << " vs ref=" << ref_loss;
}

TEST_F(SparsityOptimizerTest, ForwardPass_LargeTensor) {
    const int n = 200000;

    lfs::training::ADMMSparsityOptimizer::Config new_config;
    gs::training::ADMMSparsityOptimizer::Config ref_config;
    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    auto new_result = new_opt.compute_loss_forward(opacities);
    auto ref_result = ref_opt.compute_loss_forward(to_torch(opacities));

    ASSERT_TRUE(new_result.has_value());
    ASSERT_TRUE(ref_result.has_value());

    auto [new_loss, new_ctx] = *new_result;
    auto [ref_loss, ref_ctx] = *ref_result;

    EXPECT_TRUE(float_close(new_loss, ref_loss.template item<float>(), 1e-4f, 1e-5f))
        << "Loss mismatch for " << n << " Gaussians: new=" << new_loss << " vs ref=" << ref_loss;
}

// ===================================================================================
// Backward Pass Correctness Tests (Gradient Verification)
// ===================================================================================

TEST_F(SparsityOptimizerTest, BackwardPass_GradientCorrectness) {
    const int n = 10000;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .init_rho = 0.001f,
        .prune_ratio = 0.5f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .init_rho = 0.001f,
        .prune_ratio = 0.5f
    };
    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    // Forward pass
    auto new_result = new_opt.compute_loss_forward(opacities);
    auto ref_result = ref_opt.compute_loss_forward(to_torch(opacities));

    ASSERT_TRUE(new_result.has_value());
    ASSERT_TRUE(ref_result.has_value());

    auto [new_loss, new_ctx] = *new_result;
    auto [ref_loss, ref_ctx] = *ref_result;

    // Backward pass
    float grad_loss = 1.0f;

    // New implementation - write to gradient buffer
    auto grad_opacities_new = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
    auto new_backward_result = new_opt.compute_loss_backward(new_ctx, grad_loss, grad_opacities_new);
    ASSERT_TRUE(new_backward_result.has_value()) << new_backward_result.error();

    // Reference implementation
    auto ref_backward_result = ref_opt.compute_loss_backward(ref_ctx, grad_loss);
    ASSERT_TRUE(ref_backward_result.has_value()) << ref_backward_result.error();
    auto grad_opacities_ref = *ref_backward_result;

    // Compare gradients
    EXPECT_TRUE(tensors_close(grad_opacities_new, grad_opacities_ref, 1e-4f, 1e-5f))
        << "Gradients don't match!";
}

TEST_F(SparsityOptimizerTest, BackwardPass_GradientAccumulation) {
    const int n = 10000;

    lfs::training::ADMMSparsityOptimizer::Config new_config;
    gs::training::ADMMSparsityOptimizer::Config ref_config;
    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    // Initialize gradient buffer with some values
    auto grad_opacities_new = lfs::core::Tensor::ones({n, 1}, lfs::core::Device::CUDA) * 0.5f;
    auto grad_opacities_ref = to_torch(grad_opacities_new).clone();

    // Forward
    auto new_result = new_opt.compute_loss_forward(opacities);
    auto ref_result = ref_opt.compute_loss_forward(to_torch(opacities));

    ASSERT_TRUE(new_result.has_value());
    ASSERT_TRUE(ref_result.has_value());

    auto [new_loss, new_ctx] = *new_result;
    auto [ref_loss, ref_ctx] = *ref_result;

    // Backward (should accumulate into existing gradients)
    float grad_loss = 2.0f;

    auto new_backward_result = new_opt.compute_loss_backward(new_ctx, grad_loss, grad_opacities_new);
    ASSERT_TRUE(new_backward_result.has_value());

    auto ref_grad_result = ref_opt.compute_loss_backward(ref_ctx, grad_loss);
    ASSERT_TRUE(ref_grad_result.has_value());
    grad_opacities_ref += *ref_grad_result;

    // Compare accumulated gradients
    EXPECT_TRUE(tensors_close(grad_opacities_new, grad_opacities_ref, 1e-4f, 1e-5f))
        << "Accumulated gradients don't match!";
}

// ===================================================================================
// State Update Tests
// ===================================================================================

TEST_F(SparsityOptimizerTest, StateUpdate) {
    const int n = 10000;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };
    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);

    // Create torch tensor ONCE and keep it alive for both init and update
    auto ref_opacities = to_torch(opacities);

    // Initialize both with same opacities
    new_opt.initialize(opacities);
    clear_cuda_errors();  // Clear any CUDA errors from LFS implementation
    ref_opt.initialize(ref_opacities);

    // Perform state update - reuse same torch tensor
    auto new_update_result = new_opt.update_state(opacities);
    clear_cuda_errors();  // Clear any CUDA errors from LFS implementation
    auto ref_update_result = ref_opt.update_state(ref_opacities);

    ASSERT_TRUE(new_update_result.has_value()) << new_update_result.error();
    ASSERT_TRUE(ref_update_result.has_value()) << ref_update_result.error();

    // After update, loss should still be computable
    auto new_loss_result = new_opt.compute_loss_forward(opacities);
    auto ref_loss_result = ref_opt.compute_loss_forward(ref_opacities);

    ASSERT_TRUE(new_loss_result.has_value());
    ASSERT_TRUE(ref_loss_result.has_value());

    auto [new_loss, _1] = *new_loss_result;
    auto [ref_loss, _2] = *ref_loss_result;

    EXPECT_TRUE(float_close(new_loss, ref_loss.template item<float>(), 1e-4f, 1e-5f))
        << "Loss after state update: new=" << new_loss << " vs ref=" << ref_loss;
}

// ===================================================================================
// Prune Mask Tests
// ===================================================================================

TEST_F(SparsityOptimizerTest, PruneMask_Generation) {
    const int n = 10000;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .prune_ratio = 0.6f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .prune_ratio = 0.6f
    };
    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    auto ref_opacities = to_torch(opacities);

    // Get prune masks
    auto new_mask_result = new_opt.get_prune_mask(opacities);
    clear_cuda_errors();  // Clear any CUDA errors from LFS implementation
    auto ref_mask_result = ref_opt.get_prune_mask(ref_opacities);

    ASSERT_TRUE(new_mask_result.has_value()) << new_mask_result.error();
    ASSERT_TRUE(ref_mask_result.has_value()) << ref_mask_result.error();

    auto new_mask = *new_mask_result;
    auto ref_mask = *ref_mask_result;

    // Check mask properties
    // NOTE: Bool tensors need to be converted to Int32 before summing in LFS implementation
    int new_num_pruned = new_mask.to(lfs::core::DataType::Int32).sum().item<int>();
    clear_cuda_errors();  // Clear any CUDA errors from LFS sum/item
    int ref_num_pruned = ref_mask.sum().item<int>();
    int expected_pruned = static_cast<int>(new_config.prune_ratio * n);

    EXPECT_EQ(new_num_pruned, expected_pruned)
        << "New implementation prunes " << new_num_pruned << " Gaussians, expected " << expected_pruned;
    EXPECT_EQ(ref_num_pruned, expected_pruned)
        << "Ref implementation prunes " << ref_num_pruned << " Gaussians, expected " << expected_pruned;

    // Masks should select the same Gaussians (smallest opacities)
    // Note: Due to potential numerical differences in sorting, we allow small differences
    auto new_mask_vec = new_mask.cpu().to_vector();
    auto ref_mask_cpu = ref_mask.cpu();
    auto ref_mask_ptr = ref_mask_cpu.data_ptr<bool>();

    int matching = 0;
    for (int i = 0; i < n; i++) {
        if ((new_mask_vec[i] > 0.5f) == ref_mask_ptr[i]) {
            matching++;
        }
    }

    float match_ratio = static_cast<float>(matching) / n;
    EXPECT_GT(match_ratio, 0.95f) << "Only " << (match_ratio * 100) << "% of mask entries match";
}

TEST_F(SparsityOptimizerTest, NumToPrune) {
    const int n = 100000;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .prune_ratio = 0.7f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .prune_ratio = 0.7f
    };
    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);

    int new_num = new_opt.get_num_to_prune(opacities);
    int ref_num = ref_opt.get_num_to_prune(to_torch(opacities));

    int expected = static_cast<int>(new_config.prune_ratio * n);

    EXPECT_EQ(new_num, expected);
    EXPECT_EQ(ref_num, expected);
    EXPECT_EQ(new_num, ref_num);
}

// ===================================================================================
// Benchmark Tests - Realistic Workloads
// ===================================================================================

TEST_F(SparsityOptimizerTest, Benchmark_Forward_100K_Gaussians) {
    const int n = 100000;
    const int num_iterations = 100;

    lfs::training::ADMMSparsityOptimizer::Config new_config;
    gs::training::ADMMSparsityOptimizer::Config ref_config;
    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    std::cout << "\n=== Forward Pass Benchmark (100K Gaussians, " << num_iterations << " iterations) ===" << std::endl;

    // Warmup
    for (int i = 0; i < 10; i++) {
        auto [loss, ctx] = *new_opt.compute_loss_forward(opacities);
    }
    cudaDeviceSynchronize();

    // Benchmark new implementation
    auto start_new = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto result = new_opt.compute_loss_forward(opacities);
    }
    cudaDeviceSynchronize();
    auto end_new = std::chrono::high_resolution_clock::now();

    // Benchmark reference implementation
    auto ref_opacities = to_torch(opacities);
    auto start_ref = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto result = ref_opt.compute_loss_forward(ref_opacities);
    }
    cudaDeviceSynchronize();
    auto end_ref = std::chrono::high_resolution_clock::now();

    auto duration_new = std::chrono::duration_cast<std::chrono::microseconds>(end_new - start_new).count();
    auto duration_ref = std::chrono::duration_cast<std::chrono::microseconds>(end_ref - start_ref).count();

    std::cout << "New implementation: " << duration_new << " μs ("
              << std::fixed << std::setprecision(2) << (duration_new / static_cast<double>(num_iterations))
              << " μs/iter)" << std::endl;
    std::cout << "Ref implementation: " << duration_ref << " μs ("
              << (duration_ref / static_cast<double>(num_iterations)) << " μs/iter)" << std::endl;
    std::cout << "Speedup: " << (static_cast<double>(duration_ref) / duration_new) << "x" << std::endl;
}

TEST_F(SparsityOptimizerTest, Benchmark_Forward_1M_Gaussians) {
    const int n = 1000000;
    const int num_iterations = 100;

    lfs::training::ADMMSparsityOptimizer::Config new_config;
    gs::training::ADMMSparsityOptimizer::Config ref_config;
    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    std::cout << "\n=== Forward Pass Benchmark (1M Gaussians, " << num_iterations << " iterations) ===" << std::endl;

    // Warmup
    for (int i = 0; i < 10; i++) {
        auto [loss, ctx] = *new_opt.compute_loss_forward(opacities);
    }
    cudaDeviceSynchronize();

    // Benchmark new implementation
    auto start_new = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto result = new_opt.compute_loss_forward(opacities);
    }
    cudaDeviceSynchronize();
    auto end_new = std::chrono::high_resolution_clock::now();

    // Benchmark reference implementation
    auto ref_opacities = to_torch(opacities);
    auto start_ref = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto result = ref_opt.compute_loss_forward(ref_opacities);
    }
    cudaDeviceSynchronize();
    auto end_ref = std::chrono::high_resolution_clock::now();

    auto duration_new = std::chrono::duration_cast<std::chrono::microseconds>(end_new - start_new).count();
    auto duration_ref = std::chrono::duration_cast<std::chrono::microseconds>(end_ref - start_ref).count();

    std::cout << "New implementation: " << duration_new << " μs ("
              << std::fixed << std::setprecision(2) << (duration_new / static_cast<double>(num_iterations))
              << " μs/iter)" << std::endl;
    std::cout << "Ref implementation: " << duration_ref << " μs ("
              << (duration_ref / static_cast<double>(num_iterations)) << " μs/iter)" << std::endl;
    std::cout << "Speedup: " << (static_cast<double>(duration_ref) / duration_new) << "x" << std::endl;
}

TEST_F(SparsityOptimizerTest, Benchmark_FullPipeline_1M_Gaussians) {
    // Realistic training scenario: 1M Gaussians with forward + backward
    const int n = 1000000;
    const int num_iterations = 50;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };

    std::cout << "\n=== Full Pipeline Benchmark (1M Gaussians, " << num_iterations << " iterations) ===" << std::endl;
    std::cout << "Initializing..." << std::flush;

    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    std::cout << " done." << std::endl;

    // Warmup
    auto grad_opacities = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
    for (int i = 0; i < 5; i++) {
        auto [loss, ctx] = *new_opt.compute_loss_forward(opacities);
        grad_opacities.zero_();
        new_opt.compute_loss_backward(ctx, 1.0f, grad_opacities);
    }
    cudaDeviceSynchronize();

    std::cout << "Benchmarking new implementation..." << std::flush;

    // Benchmark new implementation (forward + backward) with detailed timing
    long long fwd_time = 0, bwd_time = 0, zero_time = 0;

    for (int i = 0; i < num_iterations; i++) {
        // Forward
        auto start_fwd = std::chrono::high_resolution_clock::now();
        auto [loss, ctx] = *new_opt.compute_loss_forward(opacities);
        cudaDeviceSynchronize();
        auto end_fwd = std::chrono::high_resolution_clock::now();
        fwd_time += std::chrono::duration_cast<std::chrono::microseconds>(end_fwd - start_fwd).count();

        // Zero gradient buffer
        auto start_zero = std::chrono::high_resolution_clock::now();
        grad_opacities.zero_();
        cudaDeviceSynchronize();
        auto end_zero = std::chrono::high_resolution_clock::now();
        zero_time += std::chrono::duration_cast<std::chrono::microseconds>(end_zero - start_zero).count();

        // Backward
        auto start_bwd = std::chrono::high_resolution_clock::now();
        new_opt.compute_loss_backward(ctx, 1.0f, grad_opacities);
        cudaDeviceSynchronize();
        auto end_bwd = std::chrono::high_resolution_clock::now();
        bwd_time += std::chrono::duration_cast<std::chrono::microseconds>(end_bwd - start_bwd).count();
    }

    auto start_new = std::chrono::high_resolution_clock::now();
    auto end_new = start_new + std::chrono::microseconds(fwd_time + bwd_time + zero_time);

    std::cout << " done." << std::endl;
    std::cout << "  Forward:  " << (fwd_time / num_iterations) << " μs/iter" << std::endl;
    std::cout << "  Zero:     " << (zero_time / num_iterations) << " μs/iter" << std::endl;
    std::cout << "  Backward: " << (bwd_time / num_iterations) << " μs/iter" << std::endl;

    // Warmup reference implementation
    auto ref_opacities = to_torch(opacities);
    for (int i = 0; i < 5; i++) {
        auto [loss, ctx] = *ref_opt.compute_loss_forward(ref_opacities);
        auto grad = ref_opt.compute_loss_backward(ctx, 1.0f);
    }
    cudaDeviceSynchronize();

    std::cout << "Benchmarking reference implementation..." << std::flush;

    // Benchmark reference implementation (forward + backward)
    auto start_ref = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        // Forward
        auto [loss, ctx] = *ref_opt.compute_loss_forward(ref_opacities);

        // Backward
        auto grad = ref_opt.compute_loss_backward(ctx, 1.0f);
    }
    cudaDeviceSynchronize();
    auto end_ref = std::chrono::high_resolution_clock::now();

    std::cout << " done." << std::endl;

    auto duration_new = std::chrono::duration_cast<std::chrono::milliseconds>(end_new - start_new).count();
    auto duration_ref = std::chrono::duration_cast<std::chrono::milliseconds>(end_ref - start_ref).count();

    std::cout << "\nResults (" << num_iterations << " iterations):" << std::endl;
    std::cout << "New implementation: " << duration_new << " ms ("
              << std::fixed << std::setprecision(2) << (duration_new / static_cast<double>(num_iterations))
              << " ms/iter)" << std::endl;
    std::cout << "Ref implementation: " << duration_ref << " ms ("
              << (duration_ref / static_cast<double>(num_iterations)) << " ms/iter)" << std::endl;
    std::cout << "Speedup: " << (static_cast<double>(duration_ref) / duration_new) << "x" << std::endl;
}

TEST_F(SparsityOptimizerTest, Benchmark_StateUpdate_1M_Gaussians) {
    const int n = 1000000;
    const int num_iterations = 50;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };

    std::cout << "\n=== State Update Benchmark (1M Gaussians, " << num_iterations << " iterations) ===" << std::endl;

    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    // Benchmark new implementation
    auto start_new = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        new_opt.update_state(opacities);
    }
    cudaDeviceSynchronize();
    auto end_new = std::chrono::high_resolution_clock::now();

    // Benchmark reference implementation
    auto ref_opacities = to_torch(opacities);
    auto start_ref = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        ref_opt.update_state(ref_opacities);
    }
    cudaDeviceSynchronize();
    auto end_ref = std::chrono::high_resolution_clock::now();

    auto duration_new = std::chrono::duration_cast<std::chrono::milliseconds>(end_new - start_new).count();
    auto duration_ref = std::chrono::duration_cast<std::chrono::milliseconds>(end_ref - start_ref).count();

    std::cout << "Results (" << num_iterations << " iterations):" << std::endl;
    std::cout << "New implementation: " << duration_new << " ms ("
              << std::fixed << std::setprecision(2) << (duration_new / static_cast<double>(num_iterations))
              << " ms/iter)" << std::endl;
    std::cout << "Ref implementation: " << duration_ref << " ms ("
              << (duration_ref / static_cast<double>(num_iterations)) << " ms/iter)" << std::endl;
    std::cout << "Speedup: " << (static_cast<double>(duration_ref) / duration_new) << "x" << std::endl;
}

TEST_F(SparsityOptimizerTest, Benchmark_PruneMask_1M_Gaussians) {
    const int n = 1000000;
    const int num_iterations = 10;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .prune_ratio = 0.6f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .prune_ratio = 0.6f
    };

    std::cout << "\n=== Prune Mask Generation Benchmark (1M Gaussians, " << num_iterations << " iterations) ===" << std::endl;

    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);

    // Benchmark new implementation
    auto start_new = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto mask = new_opt.get_prune_mask(opacities);
    }
    cudaDeviceSynchronize();
    auto end_new = std::chrono::high_resolution_clock::now();

    // Benchmark reference implementation
    auto ref_opacities = to_torch(opacities);
    auto start_ref = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto mask = ref_opt.get_prune_mask(ref_opacities);
    }
    cudaDeviceSynchronize();
    auto end_ref = std::chrono::high_resolution_clock::now();

    auto duration_new = std::chrono::duration_cast<std::chrono::milliseconds>(end_new - start_new).count();
    auto duration_ref = std::chrono::duration_cast<std::chrono::milliseconds>(end_ref - start_ref).count();

    std::cout << "Results (" << num_iterations << " iterations):" << std::endl;
    std::cout << "New implementation: " << duration_new << " ms ("
              << std::fixed << std::setprecision(2) << (duration_new / static_cast<double>(num_iterations))
              << " ms/iter)" << std::endl;
    std::cout << "Ref implementation: " << duration_ref << " ms ("
              << (duration_ref / static_cast<double>(num_iterations)) << " ms/iter)" << std::endl;
    std::cout << "Speedup: " << (static_cast<double>(duration_ref) / duration_new) << "x" << std::endl;
}

TEST_F(SparsityOptimizerTest, Benchmark_Backward_1M_Gaussians) {
    const int n = 1000000;
    const int num_iterations = 100;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };

    std::cout << "\n=== Backward Pass Benchmark (1M Gaussians, " << num_iterations << " iterations) ===" << std::endl;

    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    // Do forward pass to get contexts
    auto [loss_new, ctx_new] = *new_opt.compute_loss_forward(opacities);
    auto ref_opacities = to_torch(opacities);
    auto [loss_ref, ctx_ref] = *ref_opt.compute_loss_forward(ref_opacities);

    // Warmup
    auto grad_opacities = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
    for (int i = 0; i < 10; i++) {
        grad_opacities.zero_();
        new_opt.compute_loss_backward(ctx_new, 1.0f, grad_opacities);
    }
    cudaDeviceSynchronize();

    // Benchmark new implementation (backward only)
    auto start_new = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        grad_opacities.zero_();
        new_opt.compute_loss_backward(ctx_new, 1.0f, grad_opacities);
    }
    cudaDeviceSynchronize();
    auto end_new = std::chrono::high_resolution_clock::now();

    // Benchmark reference implementation (backward only)
    for (int i = 0; i < 10; i++) {
        auto grad = ref_opt.compute_loss_backward(ctx_ref, 1.0f);
    }
    cudaDeviceSynchronize();

    auto start_ref = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto grad = ref_opt.compute_loss_backward(ctx_ref, 1.0f);
    }
    cudaDeviceSynchronize();
    auto end_ref = std::chrono::high_resolution_clock::now();

    auto duration_new = std::chrono::duration_cast<std::chrono::microseconds>(end_new - start_new).count();
    auto duration_ref = std::chrono::duration_cast<std::chrono::microseconds>(end_ref - start_ref).count();

    std::cout << "New implementation: " << duration_new << " μs ("
              << std::fixed << std::setprecision(2) << (duration_new / static_cast<double>(num_iterations))
              << " μs/iter)" << std::endl;
    std::cout << "Ref implementation: " << duration_ref << " μs ("
              << (duration_ref / static_cast<double>(num_iterations)) << " μs/iter)" << std::endl;
    std::cout << "Speedup: " << (static_cast<double>(duration_ref) / duration_new) << "x" << std::endl;
}

// ===================================================================================
// Comprehensive Correctness Tests
// ===================================================================================

TEST_F(SparsityOptimizerTest, GradientNumericalAccuracy) {
    // Verify gradients using finite difference method
    const int n = 1000;
    const float epsilon = 1e-4f;

    lfs::training::ADMMSparsityOptimizer::Config config{
        .init_rho = 0.001f,
        .prune_ratio = 0.5f
    };
    lfs::training::ADMMSparsityOptimizer optimizer(config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA) * 0.5f;
    optimizer.initialize(opacities);

    // Forward pass
    auto [loss, ctx] = *optimizer.compute_loss_forward(opacities);

    // Compute analytical gradients
    auto grad_analytical = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
    optimizer.compute_loss_backward(ctx, 1.0f, grad_analytical);

    // Compute numerical gradients for a few random indices
    auto grad_analytical_vec = grad_analytical.cpu().to_vector();
    auto opacities_vec = opacities.cpu().to_vector();

    std::vector<int> test_indices = {0, n/4, n/2, 3*n/4, n-1};
    for (int idx : test_indices) {
        // Forward difference
        opacities_vec[idx] += epsilon;
        auto opacities_plus = lfs::core::Tensor::from_vector(opacities_vec, opacities.shape(), lfs::core::Device::CUDA);
        auto [loss_plus, _] = *optimizer.compute_loss_forward(opacities_plus);

        opacities_vec[idx] -= 2 * epsilon;
        auto opacities_minus = lfs::core::Tensor::from_vector(opacities_vec, opacities.shape(), lfs::core::Device::CUDA);
        auto [loss_minus, __] = *optimizer.compute_loss_forward(opacities_minus);

        opacities_vec[idx] += epsilon; // Restore

        float numerical_grad = (loss_plus - loss_minus) / (2 * epsilon);
        float analytical_grad = grad_analytical_vec[idx];

        EXPECT_TRUE(float_close(numerical_grad, analytical_grad, 1e-2f, 1e-3f))
            << "Index " << idx << ": numerical=" << numerical_grad
            << ", analytical=" << analytical_grad
            << ", diff=" << std::abs(numerical_grad - analytical_grad);
    }
}

TEST_F(SparsityOptimizerTest, MultipleForwardBackwardCycles) {
    // Test stability over multiple forward/backward cycles
    const int n = 5000;
    const int num_cycles = 20;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };

    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    auto ref_opacities = to_torch(opacities);

    // Run multiple cycles and verify consistency
    for (int cycle = 0; cycle < num_cycles; cycle++) {
        // Forward pass
        auto [new_loss, new_ctx] = *new_opt.compute_loss_forward(opacities);
        auto [ref_loss, ref_ctx] = *ref_opt.compute_loss_forward(ref_opacities);

        EXPECT_TRUE(float_close(new_loss, ref_loss.template item<float>(), 1e-4f, 1e-5f))
            << "Cycle " << cycle << ": loss mismatch - new=" << new_loss << ", ref=" << ref_loss;

        // Backward pass
        auto new_grad = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
        new_opt.compute_loss_backward(new_ctx, 1.0f, new_grad);

        auto ref_grad = *ref_opt.compute_loss_backward(ref_ctx, 1.0f);

        EXPECT_TRUE(tensors_close(new_grad, ref_grad, 1e-4f, 1e-5f))
            << "Cycle " << cycle << ": gradient mismatch";

        // Small update to opacities (simulate optimizer step)
        opacities = opacities - new_grad * 0.001f;
        ref_opacities = ref_opacities - ref_grad * 0.001f;
    }
}

TEST_F(SparsityOptimizerTest, ADMMVariableConsistency) {
    // Verify ADMM variables (u, z) are updated correctly
    const int n = 2000;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .init_rho = 0.001f,
        .prune_ratio = 0.5f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .init_rho = 0.001f,
        .prune_ratio = 0.5f
    };

    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    // Perform state update
    new_opt.update_state(opacities);
    ref_opt.update_state(to_torch(opacities));

    // Forward pass to verify state
    auto [new_loss, _] = *new_opt.compute_loss_forward(opacities);
    auto [ref_loss, __] = *ref_opt.compute_loss_forward(to_torch(opacities));

    EXPECT_TRUE(float_close(new_loss, ref_loss.template item<float>(), 1e-4f, 1e-5f))
        << "Loss mismatch after state update: new=" << new_loss << ", ref=" << ref_loss;
}

TEST_F(SparsityOptimizerTest, EdgeCase_AllZeros) {
    const int n = 100;

    lfs::training::ADMMSparsityOptimizer::Config config;
    lfs::training::ADMMSparsityOptimizer optimizer(config);

    auto opacities = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);

    EXPECT_TRUE(optimizer.initialize(opacities).has_value());

    auto [loss, ctx] = *optimizer.compute_loss_forward(opacities);
    EXPECT_TRUE(std::isfinite(loss));

    auto grad = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
    EXPECT_TRUE(optimizer.compute_loss_backward(ctx, 1.0f, grad).has_value());

    // Gradient should be zero or very small for all-zero input
    float grad_norm = grad.norm(2.0f);
    EXPECT_LT(grad_norm, 1e-3f) << "Gradient norm too large for all-zero input: " << grad_norm;
}

TEST_F(SparsityOptimizerTest, EdgeCase_AllOnes) {
    const int n = 100;

    lfs::training::ADMMSparsityOptimizer::Config config;
    lfs::training::ADMMSparsityOptimizer optimizer(config);

    auto opacities = lfs::core::Tensor::ones({n, 1}, lfs::core::Device::CUDA) * 10.0f; // Large positive = sigmoid→1

    EXPECT_TRUE(optimizer.initialize(opacities).has_value());

    auto [loss, ctx] = *optimizer.compute_loss_forward(opacities);
    EXPECT_TRUE(std::isfinite(loss));

    auto grad = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
    EXPECT_TRUE(optimizer.compute_loss_backward(ctx, 1.0f, grad).has_value());

    // Verify no NaN or Inf in gradients
    auto grad_vec = grad.cpu().to_vector();
    for (float g : grad_vec) {
        EXPECT_TRUE(std::isfinite(g)) << "Non-finite gradient detected: " << g;
    }
}

TEST_F(SparsityOptimizerTest, EdgeCase_SingleElement) {
    const int n = 1;

    lfs::training::ADMMSparsityOptimizer::Config config{
        .prune_ratio = 0.5f
    };
    lfs::training::ADMMSparsityOptimizer optimizer(config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);

    EXPECT_TRUE(optimizer.initialize(opacities).has_value());

    auto [loss, ctx] = *optimizer.compute_loss_forward(opacities);
    EXPECT_TRUE(std::isfinite(loss));

    auto grad = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
    EXPECT_TRUE(optimizer.compute_loss_backward(ctx, 1.0f, grad).has_value());
}

TEST_F(SparsityOptimizerTest, DifferentPruningRatios) {
    // Test various pruning ratios
    const int n = 1000;
    std::vector<float> ratios = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);

    for (float ratio : ratios) {
        lfs::training::ADMMSparsityOptimizer::Config new_config{
            .prune_ratio = ratio
        };
        gs::training::ADMMSparsityOptimizer::Config ref_config{
            .prune_ratio = ratio
        };

        lfs::training::ADMMSparsityOptimizer new_opt(new_config);
        gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

        new_opt.initialize(opacities);
        ref_opt.initialize(to_torch(opacities));

        auto new_mask_result = new_opt.get_prune_mask(opacities);
        ASSERT_TRUE(new_mask_result.has_value()) << "Failed for ratio " << ratio;

        auto ref_mask_result = ref_opt.get_prune_mask(to_torch(opacities));
        ASSERT_TRUE(ref_mask_result.has_value()) << "Failed for ratio " << ratio;

        auto new_mask = *new_mask_result;
        auto ref_mask = *ref_mask_result;

        // Count pruned elements
        int new_count = new_mask.to(lfs::core::DataType::Int32).sum().item<int>();
        int expected = static_cast<int>(ratio * n);

        EXPECT_EQ(new_count, expected)
            << "Ratio " << ratio << ": expected " << expected << " pruned, got " << new_count;
    }
}

TEST_F(SparsityOptimizerTest, StatePersistenceAcrossUpdates) {
    // Verify state is properly maintained across multiple updates
    const int n = 3000;
    const int num_updates = 10;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f
    };

    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    auto ref_opacities = to_torch(opacities);

    for (int i = 0; i < num_updates; i++) {
        // Update state
        EXPECT_TRUE(new_opt.update_state(opacities).has_value());
        EXPECT_TRUE(ref_opt.update_state(ref_opacities).has_value());

        // Verify forward pass consistency after update
        auto [new_loss, _] = *new_opt.compute_loss_forward(opacities);
        auto [ref_loss, __] = *ref_opt.compute_loss_forward(ref_opacities);

        EXPECT_TRUE(float_close(new_loss, ref_loss.template item<float>(), 1e-4f, 1e-5f))
            << "Update " << i << ": loss mismatch - new=" << new_loss << ", ref=" << ref_loss;

        // Small modification to opacities for next iteration (use same noise for both!)
        auto noise = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA) * 0.01f;
        opacities = opacities + noise;
        ref_opacities = ref_opacities + to_torch(noise);
    }
}

TEST_F(SparsityOptimizerTest, FullTrainingSimulation) {
    // Simulate a realistic training scenario
    const int n = 10000;
    const int num_iterations = 50;
    const int update_every = 10;

    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f,
        .update_every = update_every
    };
    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f,
        .update_every = update_every
    };

    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    copy_state(new_opt, ref_opt, opacities);

    auto ref_opacities = to_torch(opacities);

    for (int iter = 0; iter < num_iterations; iter++) {
        // Forward
        auto [new_loss, new_ctx] = *new_opt.compute_loss_forward(opacities);
        auto [ref_loss, ref_ctx] = *ref_opt.compute_loss_forward(ref_opacities);

        EXPECT_TRUE(float_close(new_loss, ref_loss.template item<float>(), 1e-4f, 1e-5f))
            << "Iter " << iter << ": forward loss mismatch";

        // Backward
        auto new_grad = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
        new_opt.compute_loss_backward(new_ctx, 1.0f, new_grad);

        auto ref_grad = *ref_opt.compute_loss_backward(ref_ctx, 1.0f);

        EXPECT_TRUE(tensors_close(new_grad, ref_grad, 1e-4f, 1e-5f))
            << "Iter " << iter << ": backward gradient mismatch";

        // Update opacities (simulate optimizer step)
        opacities = opacities - new_grad * 0.01f;
        ref_opacities = ref_opacities - ref_grad * 0.01f;

        // Periodic state update
        if (iter % update_every == 0) {
            new_opt.update_state(opacities);
            ref_opt.update_state(ref_opacities);
        }
    }
}

TEST_F(SparsityOptimizerTest, NumericalStability_ExtremeValues) {
    const int n = 500;

    lfs::training::ADMMSparsityOptimizer::Config config;
    lfs::training::ADMMSparsityOptimizer optimizer(config);

    // Test with very large positive values (sigmoid → 1)
    auto large_pos = lfs::core::Tensor::ones({n, 1}, lfs::core::Device::CUDA) * 100.0f;
    EXPECT_TRUE(optimizer.initialize(large_pos).has_value());

    auto [loss1, ctx1] = *optimizer.compute_loss_forward(large_pos);
    EXPECT_TRUE(std::isfinite(loss1)) << "Loss not finite for large positive values";

    auto grad1 = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
    EXPECT_TRUE(optimizer.compute_loss_backward(ctx1, 1.0f, grad1).has_value());

    // Test with very large negative values (sigmoid → 0)
    auto large_neg = lfs::core::Tensor::ones({n, 1}, lfs::core::Device::CUDA) * -100.0f;
    auto [loss2, ctx2] = *optimizer.compute_loss_forward(large_neg);
    EXPECT_TRUE(std::isfinite(loss2)) << "Loss not finite for large negative values";

    auto grad2 = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
    EXPECT_TRUE(optimizer.compute_loss_backward(ctx2, 1.0f, grad2).has_value());

    // Verify no NaN/Inf in gradients
    auto grad1_vec = grad1.cpu().to_vector();
    auto grad2_vec = grad2.cpu().to_vector();
    for (size_t i = 0; i < grad1_vec.size(); i++) {
        EXPECT_TRUE(std::isfinite(grad1_vec[i]));
        EXPECT_TRUE(std::isfinite(grad2_vec[i]));
    }
}

TEST_F(SparsityOptimizerTest, GradientAccumulationMultipleLosses) {
    // Test that gradients accumulate correctly when multiple losses contribute
    const int n = 2000;

    lfs::training::ADMMSparsityOptimizer::Config config;
    lfs::training::ADMMSparsityOptimizer optimizer(config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    optimizer.initialize(opacities);

    // First loss contribution
    auto [loss1, ctx1] = *optimizer.compute_loss_forward(opacities);
    auto grad = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
    optimizer.compute_loss_backward(ctx1, 0.5f, grad); // weight = 0.5

    // Second loss contribution (accumulate)
    auto [loss2, ctx2] = *optimizer.compute_loss_forward(opacities);
    optimizer.compute_loss_backward(ctx2, 0.3f, grad); // weight = 0.3

    // Third loss contribution (accumulate)
    auto [loss3, ctx3] = *optimizer.compute_loss_forward(opacities);
    optimizer.compute_loss_backward(ctx3, 0.2f, grad); // weight = 0.2

    // Compute expected gradient (all at once)
    auto grad_expected = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);
    optimizer.compute_loss_backward(ctx1, 1.0f, grad_expected); // weight = 1.0

    // grad should equal grad_expected since 0.5 + 0.3 + 0.2 = 1.0
    EXPECT_TRUE(tensors_close(grad, to_torch(grad_expected), 1e-4f, 1e-5f))
        << "Gradient accumulation failed";
}

TEST_F(SparsityOptimizerTest, PruneMaskConsistency) {
    // Verify pruning mask is deterministic for same input
    const int n = 5000;

    lfs::training::ADMMSparsityOptimizer::Config config{
        .prune_ratio = 0.5f
    };
    lfs::training::ADMMSparsityOptimizer optimizer(config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);

    // Generate mask multiple times
    auto mask1 = *optimizer.get_prune_mask(opacities);
    auto mask2 = *optimizer.get_prune_mask(opacities);
    auto mask3 = *optimizer.get_prune_mask(opacities);

    // All masks should be identical
    EXPECT_TRUE(tensors_close(mask1, to_torch(mask2), 0.0f, 0.0f));
    EXPECT_TRUE(tensors_close(mask1, to_torch(mask3), 0.0f, 0.0f));

    // Verify correct number of elements pruned
    int pruned_count = mask1.to(lfs::core::DataType::Int32).sum().item<int>();
    int expected_count = static_cast<int>(config.prune_ratio * n);
    EXPECT_EQ(pruned_count, expected_count);
}

TEST_F(SparsityOptimizerTest, InitializationIdempotency) {
    // Verify calling initialize multiple times doesn't break state
    const int n = 1000;

    lfs::training::ADMMSparsityOptimizer::Config config;
    lfs::training::ADMMSparsityOptimizer optimizer(config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);

    // Initialize multiple times
    EXPECT_TRUE(optimizer.initialize(opacities).has_value());
    auto [loss1, _] = *optimizer.compute_loss_forward(opacities);

    EXPECT_TRUE(optimizer.initialize(opacities).has_value());
    auto [loss2, __] = *optimizer.compute_loss_forward(opacities);

    EXPECT_TRUE(optimizer.initialize(opacities).has_value());
    auto [loss3, ___] = *optimizer.compute_loss_forward(opacities);

    // Loss should be consistent
    EXPECT_TRUE(float_close(loss1, loss2, 1e-4f, 1e-5f));
    EXPECT_TRUE(float_close(loss2, loss3, 1e-4f, 1e-5f));
}

TEST_F(SparsityOptimizerTest, ZeroGradientWeightProducesZeroGrad) {
    // Verify that grad_loss=0 produces zero gradients
    const int n = 1000;

    lfs::training::ADMMSparsityOptimizer::Config config;
    lfs::training::ADMMSparsityOptimizer optimizer(config);

    auto opacities = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    optimizer.initialize(opacities);

    auto [loss, ctx] = *optimizer.compute_loss_forward(opacities);
    auto grad = lfs::core::Tensor::zeros({n, 1}, lfs::core::Device::CUDA);

    // Backward with zero weight
    optimizer.compute_loss_backward(ctx, 0.0f, grad);

    // Gradient should be zero
    float grad_norm = grad.norm(2.0f);
    EXPECT_LT(grad_norm, 1e-6f) << "Gradient norm should be zero for grad_loss=0, got " << grad_norm;
}

} // namespace
