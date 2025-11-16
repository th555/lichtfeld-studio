/* Test to prove whether LibTorch explodes with naive split implementation */
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>

void print_memory_usage(const std::string& label) {
    // Get CUDA device memory info
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    float cuda_used_gb = (total_mem - free_mem) / (1024.0f * 1024.0f * 1024.0f);
    float cuda_free_gb = free_mem / (1024.0f * 1024.0f * 1024.0f);
    float cuda_total_gb = total_mem / (1024.0f * 1024.0f * 1024.0f);

    printf("[%s]\n", label.c_str());
    printf("  CUDA: used=%.2f GB, free=%.2f GB, total=%.2f GB\n",
           cuda_used_gb, cuda_free_gb, cuda_total_gb);

    // Get PyTorch caching allocator memory stats
    // Note: These show PyTorch's internal memory management
    try {
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);

        // StatArray is std::array<Stat, 3> with indices: AGGREGATE=0, SMALL_POOL=1, LARGE_POOL=2
        // Stat has: int64_t current, peak, allocated, freed
        float allocated_current = stats.allocated_bytes[0].current / (1024.0f * 1024.0f * 1024.0f);
        float allocated_peak = stats.allocated_bytes[0].peak / (1024.0f * 1024.0f * 1024.0f);
        float reserved_current = stats.reserved_bytes[0].current / (1024.0f * 1024.0f * 1024.0f);
        float reserved_peak = stats.reserved_bytes[0].peak / (1024.0f * 1024.0f * 1024.0f);
        float cached = (stats.reserved_bytes[0].current - stats.allocated_bytes[0].current) / (1024.0f * 1024.0f * 1024.0f);

        printf("  PyTorch Allocator:\n");
        printf("    - Allocated (current): %.2f GB (tensor data PyTorch thinks is in use)\n", allocated_current);
        printf("    - Allocated (peak):    %.2f GB\n", allocated_peak);
        printf("    - Reserved (current):  %.2f GB (memory PyTorch requested from CUDA)\n", reserved_current);
        printf("    - Reserved (peak):     %.2f GB\n", reserved_peak);
        printf("    - Cached (available):  %.2f GB (reserved - allocated = REUSABLE)\n", cached);
    } catch (...) {
        printf("  PyTorch Allocator: (not initialized yet)\n");
    }
}

TEST(LibTorchNaiveSplit, ExplodesLikeCustomTensor) {
    std::cout << "\n=== LIBTORCH NAIVE SPLIT TEST (10M GAUSSIANS) ===" << std::endl;
    std::cout << "This test replicates the EXACT naive implementation from custom Tensor" << std::endl;
    std::cout << "If LibTorch also explodes, the issue is ALGORITHMIC (too many intermediates)" << std::endl;
    std::cout << "If LibTorch doesn't explode, the issue is our Tensor implementation" << std::endl;

    const int N = 10000000;  // 10M Gaussians
    const int split_size = 2;

    print_memory_usage("BEFORE allocation");

    // Create initial data (like SplatData)
    auto means = torch::randn({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto scales3d = torch::randn({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto rotation = torch::zeros({N, 4}, torch::TensorOptions().device(torch::kCUDA));
    rotation.slice(1, 0, 1).fill_(1.0f);  // w=1
    auto sh = torch::randn({N, 48}, torch::TensorOptions().device(torch::kCUDA));
    auto opacity = torch::randn({N, 1}, torch::TensorOptions().device(torch::kCUDA));

    print_memory_usage("AFTER initial data");

    // Create selection mask (select ~99.998% for split, like the test)
    auto is_split = torch::ones({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    is_split.slice(0, 0, 2).fill_(false);  // Leave 2 not split

    // Get indices
    auto sampled_idxs = is_split.nonzero().squeeze(-1);
    auto rest_idxs = is_split.logical_not().nonzero().squeeze(-1);

    std::cout << "\nSelected for split: " << sampled_idxs.size(0) << " Gaussians" << std::endl;
    std::cout << "Rest (not split): " << rest_idxs.size(0) << " Gaussians" << std::endl;

    print_memory_usage("AFTER indices");

    // =========================================================================
    // START OF NAIVE IMPLEMENTATION (same as custom Tensor)
    // =========================================================================

    std::cout << "\n--- Starting naive split implementation ---" << std::endl;

    // 1. Index select ALL parameters upfront (like custom Tensor does)
    std::cout << "1. Index selecting all parameters..." << std::endl;
    auto sampled_means = means.index_select(0, sampled_idxs);
    auto sampled_scales = scales3d.index_select(0, sampled_idxs);
    auto sampled_quats = rotation.index_select(0, sampled_idxs);
    auto sampled_sh = sh.index_select(0, sampled_idxs);  // 1.79 GB!
    auto sampled_opacity = opacity.index_select(0, sampled_idxs);

    print_memory_usage("AFTER index_select all params");

    // 2. Extract quaternion components (like custom Tensor does)
    std::cout << "2. Extracting quaternion components..." << std::endl;
    auto w = sampled_quats.slice(1, 0, 1);
    auto x = sampled_quats.slice(1, 1, 2);
    auto y = sampled_quats.slice(1, 2, 3);
    auto z = sampled_quats.slice(1, 3, 4);

    print_memory_usage("AFTER quaternion extraction");

    // 3. Build rotation matrix element by element (like custom Tensor does)
    std::cout << "3. Building rotation matrix elements..." << std::endl;
    auto r00 = (1.0f - 2.0f * (y*y + z*z)).squeeze(-1);
    auto r01 = (2.0f * (x*y - w*z)).squeeze(-1);
    auto r02 = (2.0f * (x*z + w*y)).squeeze(-1);
    auto r10 = (2.0f * (x*y + w*z)).squeeze(-1);
    auto r11 = (1.0f - 2.0f * (x*x + z*z)).squeeze(-1);
    auto r12 = (2.0f * (y*z - w*x)).squeeze(-1);
    auto r20 = (2.0f * (x*z - w*y)).squeeze(-1);
    auto r21 = (2.0f * (y*z + w*x)).squeeze(-1);
    auto r22 = (1.0f - 2.0f * (x*x + y*y)).squeeze(-1);

    print_memory_usage("AFTER rotation matrix elements");

    // 4. Stack into rows, then matrix (like custom Tensor does)
    std::cout << "4. Stacking into rotation matrix..." << std::endl;
    auto row0 = torch::stack({r00, r01, r02}, 1);
    auto row1 = torch::stack({r10, r11, r12}, 1);
    auto row2 = torch::stack({r20, r21, r22}, 1);
    auto rotmats = torch::stack({row0, row1, row2}, 1);

    print_memory_usage("AFTER rotation matrix assembly");

    // 5. Manual batch matrix multiply (like custom Tensor does)
    std::cout << "5. Manual batch matrix multiply..." << std::endl;
    auto randn = torch::randn({split_size, sampled_idxs.size(0), 3},
                              torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> samples_list;
    for (int b = 0; b < split_size; ++b) {
        auto randn_b = randn[b];
        auto scaled_randn = sampled_scales * randn_b;
        auto scaled_randn_col = scaled_randn.unsqueeze(-1);
        auto rotated = torch::bmm(rotmats, scaled_randn_col).squeeze(-1);
        samples_list.push_back(rotated);
    }
    auto samples = torch::stack(samples_list, 0);

    print_memory_usage("AFTER samples creation");

    // 6. Create duplicate samples for each parameter (like custom Tensor does)
    std::cout << "6. Creating duplicate samples..." << std::endl;
    std::vector<torch::Tensor> means_vec(split_size);
    for (int b = 0; b < split_size; ++b) {
        means_vec[b] = sampled_means + samples[b];
    }
    auto samples_means = torch::cat(means_vec, 0);

    std::vector<torch::Tensor> scales_vec(split_size, sampled_scales);
    auto samples_scales = torch::cat(scales_vec, 0);

    std::vector<torch::Tensor> quats_vec(split_size, sampled_quats);
    auto samples_quats = torch::cat(quats_vec, 0);

    std::vector<torch::Tensor> sh_vec(split_size, sampled_sh);  // 3.58 GB!
    auto samples_sh = torch::cat(sh_vec, 0);

    std::vector<torch::Tensor> opacity_vec(split_size, sampled_opacity);
    auto samples_opacity = torch::cat(opacity_vec, 0);

    print_memory_usage("AFTER duplicate samples");

    // 7. Now simulate param_fn updates (updating each parameter)
    std::cout << "7. Simulating parameter updates..." << std::endl;

    // Update means
    {
        auto sampled_param = means.index_select(0, sampled_idxs);
        std::vector<torch::Tensor> vec(split_size);
        for (int b = 0; b < split_size; ++b) {
            vec[b] = sampled_param + samples[b];
        }
        auto split_param = torch::cat(vec, 0);
        auto rest_param = means.index_select(0, rest_idxs);
        auto result_means = torch::cat({rest_param, split_param}, 0);
        print_memory_usage("  AFTER means update");
    }

    // Update scales
    {
        auto sampled_param = scales3d.index_select(0, sampled_idxs);
        std::vector<torch::Tensor> vec(split_size, sampled_param);
        auto split_param = torch::cat(vec, 0);
        auto rest_param = scales3d.index_select(0, rest_idxs);
        auto result_scales = torch::cat({rest_param, split_param}, 0);
        print_memory_usage("  AFTER scales update");
    }

    // Update sh (THE BIG ONE - should cause OOM if algorithm is the issue)
    std::cout << "  Updating sh (THE BIG ONE - 1.79 GB sampled + 3.58 GB split)..." << std::endl;
    {
        auto sampled_param = sh.index_select(0, sampled_idxs);  // 1.79 GB
        print_memory_usage("    AFTER sampled_sh");

        std::vector<torch::Tensor> vec(split_size, sampled_param);  // Still 1.79 GB (shared)
        print_memory_usage("    AFTER vec creation");

        auto split_param = torch::cat(vec, 0);  // 3.58 GB!
        print_memory_usage("    AFTER split_sh cat");

        auto rest_param = sh.index_select(0, rest_idxs);
        print_memory_usage("    AFTER rest_sh");

        auto result_sh = torch::cat({rest_param, split_param}, 0);  // 3.58 GB final!
        print_memory_usage("    AFTER result_sh cat (CRITICAL POINT)");

        // If we get here without OOM, LibTorch handles it better!
        std::cout << "    ✅ SUCCESS: LibTorch handled the large allocation!" << std::endl;
    }

    print_memory_usage("FINAL");

    std::cout << "\n=== TEST COMPLETED ===" << std::endl;
    std::cout << "If you see this message, LibTorch did NOT explode!" << std::endl;
    std::cout << "This means our custom Tensor implementation has issues beyond the algorithm." << std::endl;
}
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core_new/tensor.hpp"
#include "kernels/densification_kernels.hpp"
#include <torch/torch.h>
#include <cmath>

using namespace lfs::core;
using namespace lfs::training::kernels;

class SplitKernelVsTorchTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }

    // Helper: Compare Tensor with torch::Tensor
    void expect_tensors_near(const Tensor& our_tensor, const torch::Tensor& torch_tensor,
                            const std::string& name, float tol = 1e-5f) {
        auto our_cpu = our_tensor.to(Device::CPU);
        auto torch_cpu = torch_tensor.cpu();

        ASSERT_EQ(our_cpu.numel(), torch_cpu.numel()) << "Size mismatch for " << name;

        const float* our_ptr = our_cpu.ptr<float>();
        const float* torch_ptr = torch_cpu.data_ptr<float>();

        for (size_t i = 0; i < our_cpu.numel(); ++i) {
            EXPECT_NEAR(our_ptr[i], torch_ptr[i], tol)
                << name << " mismatch at index " << i
                << ": ours=" << our_ptr[i] << ", torch=" << torch_ptr[i];
        }
    }

    // LibTorch reference implementation of split
    struct TorchSplitResult {
        torch::Tensor positions;
        torch::Tensor rotations;
        torch::Tensor scales;
        torch::Tensor sh;
        torch::Tensor opacities;
    };

    TorchSplitResult torch_split_gaussians(
        const torch::Tensor& positions_in,
        const torch::Tensor& rotations_in,
        const torch::Tensor& scales_in,
        const torch::Tensor& sh_in,
        const torch::Tensor& opacities_in,
        const torch::Tensor& split_indices,
        const torch::Tensor& keep_indices,
        const torch::Tensor& random_noise,
        bool revised_opacity = false) {

        const int64_t num_split = split_indices.size(0);
        const int64_t num_keep = keep_indices.size(0);

        // Select Gaussians to split
        auto sampled_positions = positions_in.index_select(0, split_indices);
        auto sampled_rotations = rotations_in.index_select(0, split_indices);
        auto sampled_scales = scales_in.index_select(0, split_indices);
        auto sampled_sh = sh_in.index_select(0, split_indices);
        auto sampled_opacities = opacities_in.index_select(0, split_indices);

        // Convert quaternions to rotation matrices
        auto w = sampled_rotations.select(1, 0);
        auto x = sampled_rotations.select(1, 1);
        auto y = sampled_rotations.select(1, 2);
        auto z = sampled_rotations.select(1, 3);

        auto r00 = 1.0f - 2.0f * (y*y + z*z);
        auto r01 = 2.0f * (x*y - w*z);
        auto r02 = 2.0f * (x*z + w*y);
        auto r10 = 2.0f * (x*y + w*z);
        auto r11 = 1.0f - 2.0f * (x*x + z*z);
        auto r12 = 2.0f * (y*z - w*x);
        auto r20 = 2.0f * (x*z - w*y);
        auto r21 = 2.0f * (y*z + w*x);
        auto r22 = 1.0f - 2.0f * (x*x + y*y);

        // Stack to [N, 3, 3]
        auto row0 = torch::stack({r00, r01, r02}, 1);
        auto row1 = torch::stack({r10, r11, r12}, 1);
        auto row2 = torch::stack({r20, r21, r22}, 1);
        auto rotmats = torch::stack({row0, row1, row2}, 1);

        // Compute offset samples: rotmats @ (exp(scales) * random_noise)
        // random_noise: [2, num_split, 3]
        // sampled_scales: [num_split, 3]
        auto exp_scales = sampled_scales.exp();

        // For each split copy (2 total)
        std::vector<torch::Tensor> split_positions;
        for (int s = 0; s < 2; ++s) {
            // Scale random noise
            auto noise_s = random_noise[s];  // [num_split, 3]
            auto scaled_noise = exp_scales * noise_s;  // [num_split, 3]

            // Matrix multiply: rotmats @ scaled_noise
            auto offset = torch::bmm(rotmats, scaled_noise.unsqueeze(-1)).squeeze(-1);  // [num_split, 3]

            // Position: original + offset (each copy has different offset from different random noise)
            auto new_pos = sampled_positions + offset;
            split_positions.push_back(new_pos);
        }

        // New scales: log(exp(old_scale) / 1.6) = old_scale - log(1.6)
        auto new_scales = sampled_scales - std::log(1.6f);
        std::vector<torch::Tensor> split_scales = {new_scales, new_scales};

        // Handle opacity
        torch::Tensor new_opacities;
        if (revised_opacity) {
            auto sigmoid_vals = torch::sigmoid(sampled_opacities);
            auto one_minus_sigmoid = 1.0f - sigmoid_vals;
            auto adjusted = 1.0f - one_minus_sigmoid.sqrt();
            new_opacities = torch::logit(adjusted);
        } else {
            new_opacities = sampled_opacities;
        }
        std::vector<torch::Tensor> split_opacities = {new_opacities, new_opacities};

        // Rotations and SH: just duplicate
        std::vector<torch::Tensor> split_rotations = {sampled_rotations, sampled_rotations};
        std::vector<torch::Tensor> split_sh = {sampled_sh, sampled_sh};

        // Concatenate split copies
        auto positions_split = torch::cat(split_positions, 0);
        auto rotations_split = torch::cat(split_rotations, 0);
        auto scales_split = torch::cat(split_scales, 0);
        auto sh_split = torch::cat(split_sh, 0);
        auto opacities_split = torch::cat(split_opacities, 0);

        // Get kept Gaussians
        auto positions_keep = positions_in.index_select(0, keep_indices);
        auto rotations_keep = rotations_in.index_select(0, keep_indices);
        auto scales_keep = scales_in.index_select(0, keep_indices);
        auto sh_keep = sh_in.index_select(0, keep_indices);
        auto opacities_keep = opacities_in.index_select(0, keep_indices);

        // Final concatenation: [kept, split]
        TorchSplitResult result;
        result.positions = torch::cat({positions_keep, positions_split}, 0);
        result.rotations = torch::cat({rotations_keep, rotations_split}, 0);
        result.scales = torch::cat({scales_keep, scales_split}, 0);
        result.sh = torch::cat({sh_keep, sh_split}, 0);
        result.opacities = torch::cat({opacities_keep, opacities_split}, 0);

        return result;
    }
};

#if 0  // DISABLED: Kernel signature changed to split sh0/shN - needs update
TEST_F(SplitKernelVsTorchTest, DISABLED_BasicSplit_CompareWithTorch) {
    // Test parameters
    const int N = 100;
    const int num_split = 30;
    const int num_keep = N - num_split;
    const int sh_dim = 16;

    // Create input data (on CPU first for easy initialization)
    std::vector<float> pos_data(N * 3);
    std::vector<float> rot_data(N * 4);
    std::vector<float> scale_data(N * 3);
    std::vector<float> sh_data(N * sh_dim);
    std::vector<float> opacity_data(N);

    for (int i = 0; i < N; ++i) {
        // Position
        pos_data[i * 3 + 0] = static_cast<float>(i);
        pos_data[i * 3 + 1] = static_cast<float>(i * 2);
        pos_data[i * 3 + 2] = static_cast<float>(i * 3);

        // Rotation (identity quaternion)
        rot_data[i * 4 + 0] = 1.0f;  // w
        rot_data[i * 4 + 1] = 0.0f;  // x
        rot_data[i * 4 + 2] = 0.0f;  // y
        rot_data[i * 4 + 3] = 0.0f;  // z

        // Scale (log-space, so exp(0) = 1)
        scale_data[i * 3 + 0] = 0.0f;
        scale_data[i * 3 + 1] = 0.0f;
        scale_data[i * 3 + 2] = 0.0f;

        // SH
        for (int j = 0; j < sh_dim; ++j) {
            sh_data[i * sh_dim + j] = static_cast<float>(i + j);
        }

        // Opacity
        opacity_data[i] = 0.5f;
    }

    // Split indices: first num_split
    std::vector<int64_t> split_vec(num_split);
    for (int i = 0; i < num_split; ++i) split_vec[i] = i;

    // Keep indices: remaining
    std::vector<int64_t> keep_vec(num_keep);
    for (int i = 0; i < num_keep; ++i) keep_vec[i] = num_split + i;

    // Random noise: shape [2, num_split, 3] with different values for each copy
    std::vector<float> noise_data(2 * num_split * 3);
    for (auto& n : noise_data) n = (rand() % 1000) / 1000.0f - 0.5f;

    // Create our tensors (CUDA)
    auto positions = Tensor::from_blob(pos_data.data(), TensorShape{(size_t)N, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto rotations = Tensor::from_blob(rot_data.data(), TensorShape{(size_t)N, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto scales = Tensor::from_blob(scale_data.data(), TensorShape{(size_t)N, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh = Tensor::from_blob(sh_data.data(), TensorShape{(size_t)N, (size_t)sh_dim}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto opacities = Tensor::from_blob(opacity_data.data(), TensorShape{(size_t)N}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto split_indices = Tensor::from_blob(split_vec.data(), TensorShape{(size_t)num_split}, Device::CPU, DataType::Int64).to(Device::CUDA);
    auto keep_indices = Tensor::from_blob(keep_vec.data(), TensorShape{(size_t)num_keep}, Device::CPU, DataType::Int64).to(Device::CUDA);
    auto random_noise = Tensor::from_blob(noise_data.data(), TensorShape{2, (size_t)num_split, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);

    // Create torch tensors
    auto torch_positions = torch::from_blob(pos_data.data(), {N, 3}, torch::kFloat32).clone().cuda();
    auto torch_rotations = torch::from_blob(rot_data.data(), {N, 4}, torch::kFloat32).clone().cuda();
    auto torch_scales = torch::from_blob(scale_data.data(), {N, 3}, torch::kFloat32).clone().cuda();
    auto torch_sh = torch::from_blob(sh_data.data(), {N, sh_dim}, torch::kFloat32).clone().cuda();
    auto torch_opacities = torch::from_blob(opacity_data.data(), {N}, torch::kFloat32).clone().cuda();
    auto torch_split_indices = torch::from_blob(split_vec.data(), {num_split}, torch::kInt64).clone().cuda();
    auto torch_keep_indices = torch::from_blob(keep_vec.data(), {num_keep}, torch::kInt64).clone().cuda();
    auto torch_random_noise = torch::from_blob(noise_data.data(), {2, num_split, 3}, torch::kFloat32).clone().cuda();

    // Allocate output tensors for our kernel
    const int total_out = num_keep + num_split * 2;
    auto positions_out = Tensor::empty({(size_t)total_out, 3}, Device::CUDA, DataType::Float32);
    auto rotations_out = Tensor::empty({(size_t)total_out, 4}, Device::CUDA, DataType::Float32);
    auto scales_out = Tensor::empty({(size_t)total_out, 3}, Device::CUDA, DataType::Float32);
    auto sh_out = Tensor::empty({(size_t)total_out, (size_t)sh_dim}, Device::CUDA, DataType::Float32);
    auto opacities_out = Tensor::empty({(size_t)total_out}, Device::CUDA, DataType::Float32);

    // Run our custom kernel
    launch_split_gaussians(
        positions.ptr<float>(), rotations.ptr<float>(), scales.ptr<float>(),
        sh.ptr<float>(), opacities.ptr<float>(),
        positions_out.ptr<float>(), rotations_out.ptr<float>(), scales_out.ptr<float>(),
        sh_out.ptr<float>(), opacities_out.ptr<float>(),
        split_indices.ptr<int64_t>(), keep_indices.ptr<int64_t>(),
        random_noise.ptr<float>(),
        N, num_split, num_keep, sh_dim, false, nullptr
    );

    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // Run LibTorch reference
    auto torch_result = torch_split_gaussians(
        torch_positions, torch_rotations, torch_scales, torch_sh, torch_opacities,
        torch_split_indices, torch_keep_indices, torch_random_noise, false
    );

    // Compare results
    expect_tensors_near(positions_out, torch_result.positions, "positions", 1e-4f);
    expect_tensors_near(rotations_out, torch_result.rotations, "rotations");
    expect_tensors_near(scales_out, torch_result.scales, "scales");
    expect_tensors_near(sh_out, torch_result.sh, "sh");
    expect_tensors_near(opacities_out, torch_result.opacities, "opacities");
}

// DISABLED: Kernel signature changed to split sh0/shN - needs update
TEST_F(SplitKernelVsTorchTest, DISABLED_RevisedOpacity_CompareWithTorch) {
    const int N = 50;
    const int num_split = 20;
    const int num_keep = N - num_split;
    const int sh_dim = 3;

    // Create simple test data
    std::vector<float> pos_data(N * 3, 0.0f);
    std::vector<float> rot_data(N * 4, 0.0f);
    for (int i = 0; i < N; ++i) rot_data[i * 4] = 1.0f;  // w = 1
    std::vector<float> scale_data(N * 3, 0.0f);
    std::vector<float> sh_data(N * sh_dim, 1.0f);
    std::vector<float> opacity_data(N, 0.5f);

    std::vector<int64_t> split_vec(num_split);
    for (int i = 0; i < num_split; ++i) split_vec[i] = i;
    std::vector<int64_t> keep_vec(num_keep);
    for (int i = 0; i < num_keep; ++i) keep_vec[i] = num_split + i;
    std::vector<float> noise_data(2 * num_split * 3, 0.01f);

    // Create tensors
    auto positions = Tensor::from_blob(pos_data.data(), TensorShape{(size_t)N, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto rotations = Tensor::from_blob(rot_data.data(), TensorShape{(size_t)N, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto scales = Tensor::from_blob(scale_data.data(), TensorShape{(size_t)N, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh = Tensor::from_blob(sh_data.data(), TensorShape{(size_t)N, (size_t)sh_dim}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto opacities = Tensor::from_blob(opacity_data.data(), TensorShape{(size_t)N}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto split_indices = Tensor::from_blob(split_vec.data(), TensorShape{(size_t)num_split}, Device::CPU, DataType::Int64).to(Device::CUDA);
    auto keep_indices = Tensor::from_blob(keep_vec.data(), TensorShape{(size_t)num_keep}, Device::CPU, DataType::Int64).to(Device::CUDA);
    auto random_noise = Tensor::from_blob(noise_data.data(), TensorShape{2, (size_t)num_split, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);

    auto torch_positions = torch::from_blob(pos_data.data(), {N, 3}, torch::kFloat32).clone().cuda();
    auto torch_rotations = torch::from_blob(rot_data.data(), {N, 4}, torch::kFloat32).clone().cuda();
    auto torch_scales = torch::from_blob(scale_data.data(), {N, 3}, torch::kFloat32).clone().cuda();
    auto torch_sh = torch::from_blob(sh_data.data(), {N, sh_dim}, torch::kFloat32).clone().cuda();
    auto torch_opacities = torch::from_blob(opacity_data.data(), {N}, torch::kFloat32).clone().cuda();
    auto torch_split_indices = torch::from_blob(split_vec.data(), {num_split}, torch::kInt64).clone().cuda();
    auto torch_keep_indices = torch::from_blob(keep_vec.data(), {num_keep}, torch::kInt64).clone().cuda();
    auto torch_random_noise = torch::from_blob(noise_data.data(), {2, num_split, 3}, torch::kFloat32).clone().cuda();

    const int total_out = num_keep + num_split * 2;
    auto positions_out = Tensor::empty({(size_t)total_out, 3}, Device::CUDA, DataType::Float32);
    auto rotations_out = Tensor::empty({(size_t)total_out, 4}, Device::CUDA, DataType::Float32);
    auto scales_out = Tensor::empty({(size_t)total_out, 3}, Device::CUDA, DataType::Float32);
    auto sh_out = Tensor::empty({(size_t)total_out, (size_t)sh_dim}, Device::CUDA, DataType::Float32);
    auto opacities_out = Tensor::empty({(size_t)total_out}, Device::CUDA, DataType::Float32);

    // Run with revised_opacity = true
    launch_split_gaussians(
        positions.ptr<float>(), rotations.ptr<float>(), scales.ptr<float>(),
        sh.ptr<float>(), opacities.ptr<float>(),
        positions_out.ptr<float>(), rotations_out.ptr<float>(), scales_out.ptr<float>(),
        sh_out.ptr<float>(), opacities_out.ptr<float>(),
        split_indices.ptr<int64_t>(), keep_indices.ptr<int64_t>(),
        random_noise.ptr<float>(),
        N, num_split, num_keep, sh_dim, true, nullptr  // revised_opacity = true
    );

    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    auto torch_result = torch_split_gaussians(
        torch_positions, torch_rotations, torch_scales, torch_sh, torch_opacities,
        torch_split_indices, torch_keep_indices, torch_random_noise, true
    );

    // The opacity formula is the key difference here
    expect_tensors_near(opacities_out, torch_result.opacities, "revised_opacities", 1e-5f);
}
#endif  // End of disabled split kernel tests

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core_new/tensor.hpp"
#include <torch/torch.h>

using namespace lfs::core;

class TensorFillVsTorchTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }

    // Helper: Compare our Tensor with torch::Tensor
    void expect_tensors_equal(const Tensor& our_tensor, const torch::Tensor& torch_tensor, float tol = 1e-6f) {
        auto our_cpu = our_tensor.to(Device::CPU);
        auto torch_cpu = torch_tensor.cpu();

        ASSERT_EQ(our_cpu.numel(), torch_cpu.numel());

        const float* our_ptr = our_cpu.ptr<float>();
        const float* torch_ptr = torch_cpu.data_ptr<float>();

        for (size_t i = 0; i < our_cpu.numel(); ++i) {
            EXPECT_NEAR(our_ptr[i], torch_ptr[i], tol)
                << "Mismatch at index " << i;
        }
    }
};

TEST_F(TensorFillVsTorchTest, ContiguousTensor_CPU) {
    // Create tensors
    auto our_tensor = Tensor::zeros({10, 5}, Device::CPU);
    auto torch_tensor = torch::zeros({10, 5}, torch::dtype(torch::kFloat32));

    // Fill with same value
    our_tensor.fill_(3.14f);
    torch_tensor.fill_(3.14f);

    expect_tensors_equal(our_tensor, torch_tensor);
}

TEST_F(TensorFillVsTorchTest, ContiguousTensor_CUDA) {
    // Create tensors
    auto our_tensor = Tensor::zeros({10, 5}, Device::CUDA);
    auto torch_tensor = torch::zeros({10, 5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // Fill with same value
    our_tensor.fill_(3.14f);
    torch_tensor.fill_(3.14f);

    expect_tensors_equal(our_tensor, torch_tensor);
}

TEST_F(TensorFillVsTorchTest, SlicedTensor_SingleColumn_CUDA) {
    // THIS IS THE CRITICAL BUG CASE!
    // Create [N, 4] tensor and fill only first column
    const int N = 1000;

    auto our_tensor = Tensor::zeros({N, 4}, Device::CUDA);
    auto torch_tensor = torch::zeros({N, 4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // Slice first column and fill with 1.0
    our_tensor.slice(1, 0, 1).fill_(1.0f);
    torch_tensor.slice(1, 0, 1).fill_(1.0f);

    // Check full tensor (should be [1, 0, 0, 0] for each row)
    expect_tensors_equal(our_tensor, torch_tensor);

    // Verify explicitly
    auto our_cpu = our_tensor.to(Device::CPU);
    const float* ptr = our_cpu.ptr<float>();
    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(ptr[i * 4 + 0], 1.0f) << "Row " << i << " col 0";
        EXPECT_FLOAT_EQ(ptr[i * 4 + 1], 0.0f) << "Row " << i << " col 1";
        EXPECT_FLOAT_EQ(ptr[i * 4 + 2], 0.0f) << "Row " << i << " col 2";
        EXPECT_FLOAT_EQ(ptr[i * 4 + 3], 0.0f) << "Row " << i << " col 3";
    }
}

TEST_F(TensorFillVsTorchTest, SlicedTensor_MiddleColumn_CUDA) {
    const int N = 500;

    auto our_tensor = Tensor::zeros({N, 4}, Device::CUDA);
    auto torch_tensor = torch::zeros({N, 4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // Slice column 2 and fill
    our_tensor.slice(1, 2, 3).fill_(2.5f);
    torch_tensor.slice(1, 2, 3).fill_(2.5f);

    expect_tensors_equal(our_tensor, torch_tensor);
}

TEST_F(TensorFillVsTorchTest, SlicedTensor_MultipleColumns_CUDA) {
    const int N = 500;

    auto our_tensor = Tensor::zeros({N, 4}, Device::CUDA);
    auto torch_tensor = torch::zeros({N, 4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // Slice columns 1-3 and fill
    our_tensor.slice(1, 1, 3).fill_(7.5f);
    torch_tensor.slice(1, 1, 3).fill_(7.5f);

    expect_tensors_equal(our_tensor, torch_tensor);
}

TEST_F(TensorFillVsTorchTest, SlicedTensor_Rows_CUDA) {
    auto our_tensor = Tensor::zeros({100, 5}, Device::CUDA);
    auto torch_tensor = torch::zeros({100, 5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // Slice rows 10-20 and fill
    our_tensor.slice(0, 10, 20).fill_(5.0f);
    torch_tensor.slice(0, 10, 20).fill_(5.0f);

    expect_tensors_equal(our_tensor, torch_tensor);
}

TEST_F(TensorFillVsTorchTest, DoubleSlice_CUDA) {
    auto our_tensor = Tensor::zeros({50, 10}, Device::CUDA);
    auto torch_tensor = torch::zeros({50, 10}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // Double slice: rows 5-15, columns 2-5
    our_tensor.slice(0, 5, 15).slice(1, 2, 5).fill_(9.9f);
    torch_tensor.slice(0, 5, 15).slice(1, 2, 5).fill_(9.9f);

    expect_tensors_equal(our_tensor, torch_tensor);
}

TEST_F(TensorFillVsTorchTest, TransposedTensor_CUDA) {
    auto our_tensor = Tensor::zeros({10, 5}, Device::CUDA);
    auto torch_tensor = torch::zeros({10, 5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // Transpose and fill
    auto our_t = our_tensor.t();
    auto torch_t = torch_tensor.t();

    our_t.fill_(4.2f);
    torch_t.fill_(4.2f);

    // Compare original tensors (should both be filled)
    expect_tensors_equal(our_tensor, torch_tensor);
}

TEST_F(TensorFillVsTorchTest, StressTest_LargeSlicedTensor_CUDA) {
    // Simulate the rotation quaternion case from the bug
    const int N = 10000000;  // 10M Gaussians

    std::cout << "Creating large tensors (" << N << " x 4)..." << std::endl;
    auto our_tensor = Tensor::zeros({N, 4}, Device::CUDA);
    auto torch_tensor = torch::zeros({N, 4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    std::cout << "Filling first column..." << std::endl;
    our_tensor.slice(1, 0, 1).fill_(1.0f);
    torch_tensor.slice(1, 0, 1).fill_(1.0f);

    std::cout << "Comparing results..." << std::endl;
    // Sample comparison (full comparison would be too slow)
    auto our_cpu = our_tensor.to(Device::CPU);
    auto torch_cpu = torch_tensor.cpu();

    const float* our_ptr = our_cpu.ptr<float>();
    const float* torch_ptr = torch_cpu.data_ptr<float>();

    // Check first 1000, middle 1000, last 1000
    for (int i = 0; i < 1000; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(our_ptr[i * 4 + j], torch_ptr[i * 4 + j])
                << "Mismatch at start row " << i << " col " << j;
        }
    }

    int mid = N / 2;
    for (int i = mid; i < mid + 1000; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(our_ptr[i * 4 + j], torch_ptr[i * 4 + j])
                << "Mismatch at middle row " << i << " col " << j;
        }
    }

    for (int i = N - 1000; i < N; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(our_ptr[i * 4 + j], torch_ptr[i * 4 + j])
                << "Mismatch at end row " << i << " col " << j;
        }
    }

    std::cout << "Large tensor test passed!" << std::endl;
}

TEST_F(TensorFillVsTorchTest, Int32Dtype_Sliced_CUDA) {
    auto our_tensor = Tensor::zeros({100, 3}, Device::CUDA, DataType::Int32);
    auto torch_tensor = torch::zeros({100, 3}, torch::device(torch::kCUDA).dtype(torch::kInt32));

    // Fill middle column with int value
    our_tensor.slice(1, 1, 2).fill_(42.0f);  // Will be cast to int
    torch_tensor.slice(1, 1, 2).fill_(42);

    auto our_cpu = our_tensor.to(Device::CPU);
    auto torch_cpu = torch_tensor.cpu();

    const int* our_ptr = our_cpu.ptr<int>();
    const int* torch_ptr = torch_cpu.data_ptr<int>();

    for (size_t i = 0; i < our_cpu.numel(); ++i) {
        EXPECT_EQ(our_ptr[i], torch_ptr[i]) << "Mismatch at index " << i;
    }
}

TEST_F(TensorFillVsTorchTest, BoolDtype_Sliced_CUDA) {
    auto our_tensor = Tensor::zeros_bool({100, 3}, Device::CUDA);
    auto torch_tensor = torch::zeros({100, 3}, torch::device(torch::kCUDA).dtype(torch::kBool));

    // Fill middle column with true
    our_tensor.slice(1, 1, 2).fill_(1.0f);
    torch_tensor.slice(1, 1, 2).fill_(true);

    auto our_cpu = our_tensor.to(Device::CPU);
    auto torch_cpu = torch_tensor.cpu();

    const unsigned char* our_ptr = our_cpu.ptr<unsigned char>();
    const bool* torch_ptr = torch_cpu.data_ptr<bool>();

    for (size_t i = 0; i < our_cpu.numel(); ++i) {
        EXPECT_EQ(our_ptr[i] != 0, torch_ptr[i]) << "Mismatch at index " << i;
    }
}

TEST_F(TensorFillVsTorchTest, CPU_SlicedTensor) {
    // Test CPU path for non-contiguous tensors
    auto our_tensor = Tensor::zeros({100, 4}, Device::CPU);
    auto torch_tensor = torch::zeros({100, 4}, torch::dtype(torch::kFloat32));

    our_tensor.slice(1, 0, 1).fill_(1.0f);
    torch_tensor.slice(1, 0, 1).fill_(1.0f);

    expect_tensors_equal(our_tensor, torch_tensor);
}
