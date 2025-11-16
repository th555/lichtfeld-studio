/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// LibTorch-free implementation
#include "training_new/components/sparsity_optimizer.hpp"
#include "core_new/tensor.hpp"

// Reference LibTorch implementation
#include "training/components/sparsity_optimizer.hpp"

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

TEST(SparsityDebug, DetailedComparison) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n = 100;  // Small for detailed logging

    // Create identical configs
    lfs::training::ADMMSparsityOptimizer::Config new_config{
        .sparsify_steps = 15000,
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f,
        .update_every = 50,
        .start_iteration = 30000
    };

    gs::training::ADMMSparsityOptimizer::Config ref_config{
        .sparsify_steps = 15000,
        .init_rho = 0.0005f,
        .prune_ratio = 0.6f,
        .update_every = 50,
        .start_iteration = 30000
    };

    lfs::training::ADMMSparsityOptimizer new_opt(new_config);
    gs::training::ADMMSparsityOptimizer ref_opt(ref_config);

    // Create identical input opacities
    auto opacities_lfs = lfs::core::Tensor::randn({n, 1}, lfs::core::Device::CUDA);
    auto opacities_torch = to_torch(opacities_lfs);

    std::cout << "\n=== INITIALIZATION ===" << std::endl;

    // Initialize both
    auto init_new = new_opt.initialize(opacities_lfs);
    auto init_ref = ref_opt.initialize(opacities_torch);

    ASSERT_TRUE(init_new.has_value()) << init_new.error();
    ASSERT_TRUE(init_ref.has_value()) << init_ref.error();

    std::cout << "Both initialized successfully" << std::endl;
    std::cout << "Checking input to prune_z (opa_plus_u = sigmoid(opacities) + zeros):" << std::endl;

    // The input to prune_z should be sigmoid(opacities) since u=0 initially
    auto opa_test = opacities_lfs.sigmoid();
    auto opa_test_cpu = opa_test.cpu().to_vector();
    std::cout << "First 10 opa values (should match ref): ";
    for (int i = 0; i < std::min(10, n); i++) {
        std::cout << opa_test_cpu[i] << " ";
    }
    std::cout << std::endl;

    // Now compute loss and compare internal state
    std::cout << "\n=== FORWARD PASS ===" << std::endl;

    auto new_result = new_opt.compute_loss_forward(opacities_lfs);
    auto ref_result = ref_opt.compute_loss_forward(opacities_torch);

    ASSERT_TRUE(new_result.has_value());
    ASSERT_TRUE(ref_result.has_value());

    auto [new_loss, new_ctx] = *new_result;
    auto [ref_loss, ref_ctx] = *ref_result;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "New loss: " << new_loss << std::endl;
    std::cout << "Ref loss: " << ref_loss << std::endl;
    std::cout << "Ratio: " << (new_loss / ref_loss) << std::endl;

    // Compare internal context values
    std::cout << "\n=== COMPARING CONTEXT VALUES ===" << std::endl;

    // Check opacities
    auto opacities_lfs_cpu = opacities_lfs.cpu().to_vector();
    auto opacities_torch_cpu = opacities_torch.cpu();

    std::cout << "First 5 opacities (raw):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  [" << i << "] new=" << opacities_lfs_cpu[i]
                  << ", ref=" << opacities_torch_cpu[i].template item<float>() << std::endl;
    }

    // Copy GPU data to CPU for inspection
    std::vector<float> new_opa_sigmoid_cpu(n);
    std::vector<float> new_z_cpu(n);
    std::vector<float> new_u_cpu(n);

    cudaMemcpy(new_opa_sigmoid_cpu.data(), new_ctx.opa_sigmoid_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(new_z_cpu.data(), new_ctx.z_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(new_u_cpu.data(), new_ctx.u_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);

    auto ref_opa_sigmoid_cpu = ref_ctx.opa_sigmoid.cpu();
    auto ref_z_cpu = ref_ctx.z.cpu();
    auto ref_u_cpu = ref_ctx.u.cpu();

    // Check sigmoid(opacities)
    std::cout << "\nFirst 5 sigmoid(opacities):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  [" << i << "] new=" << new_opa_sigmoid_cpu[i]
                  << ", ref=" << ref_opa_sigmoid_cpu.index({i}).item<float>() << std::endl;
    }

    // Check z values
    std::cout << "\nFirst 5 z values:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  [" << i << "] new=" << new_z_cpu[i]
                  << ", ref=" << ref_z_cpu.index({i}).item<float>() << std::endl;
    }

    // Check u values
    std::cout << "\nFirst 5 u values:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  [" << i << "] new=" << new_u_cpu[i]
                  << ", ref=" << ref_u_cpu.index({i}).item<float>() << std::endl;
    }

    // Compute diff = opa - z + u manually
    std::cout << "\n=== MANUAL DIFF CALCULATION ===" << std::endl;

    float new_diff_sum = 0.0f;
    float ref_diff_sum = 0.0f;

    for (int i = 0; i < std::min(5, n); i++) {
        float new_diff = new_opa_sigmoid_cpu[i] - new_z_cpu[i] + new_u_cpu[i];
        float ref_diff = ref_opa_sigmoid_cpu.index({i}).item<float>()
                       - ref_z_cpu.index({i}).item<float>()
                       + ref_u_cpu.index({i}).item<float>();

        std::cout << "  [" << i << "] new_diff=" << new_diff << ", ref_diff=" << ref_diff << std::endl;
    }

    // Compute full diff norm manually
    for (int i = 0; i < n; i++) {
        float new_diff = new_opa_sigmoid_cpu[i] - new_z_cpu[i] + new_u_cpu[i];
        float ref_diff = ref_opa_sigmoid_cpu.index({i}).item<float>()
                       - ref_z_cpu.index({i}).item<float>()
                       + ref_u_cpu.index({i}).item<float>();

        new_diff_sum += new_diff * new_diff;
        ref_diff_sum += ref_diff * ref_diff;
    }

    float new_diff_norm = std::sqrt(new_diff_sum);
    float ref_diff_norm = std::sqrt(ref_diff_sum);

    std::cout << "\nManual diff L2 norm:" << std::endl;
    std::cout << "  New: " << new_diff_norm << std::endl;
    std::cout << "  Ref: " << ref_diff_norm << std::endl;

    float new_manual_loss = 0.5f * new_config.init_rho * new_diff_norm * new_diff_norm;
    float ref_manual_loss = 0.5f * ref_config.init_rho * ref_diff_norm * ref_diff_norm;

    std::cout << "\nManual loss calculation:" << std::endl;
    std::cout << "  New: " << new_manual_loss << std::endl;
    std::cout << "  Ref: " << ref_manual_loss << std::endl;
    std::cout << "  Reported new: " << new_loss << std::endl;
    std::cout << "  Reported ref: " << ref_loss << std::endl;
}
