/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

// LibTorch-free implementation
#include "training_new/components/bilateral_grid.hpp"
#include "core_new/tensor.hpp"

// Reference LibTorch implementation
#include "training/components/bilateral_grid.hpp"

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

// Copy grids from reference to new implementation for fair comparison
void copy_grids(lfs::training::BilateralGrid& new_grid, gs::training::BilateralGrid& ref_grid) {
    auto ref_params = ref_grid.parameters();
    auto new_params_tensor = from_torch(ref_params);

    // Copy data directly
    cudaMemcpy(new_grid.parameters().ptr<float>(),
               new_params_tensor.ptr<float>(),
               new_params_tensor.numel() * sizeof(float),
               cudaMemcpyDeviceToDevice);
}

} // namespace

// ===================================================================================
// Basic Functionality Tests
// ===================================================================================

class BilateralGridTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available, skipping test";
        }
    }
};

TEST_F(BilateralGridTest, Construction) {
    const int num_images = 10;
    const int grid_W = 16, grid_H = 16, grid_L = 8;

    lfs::training::BilateralGrid new_grid(num_images, grid_W, grid_H, grid_L);
    gs::training::BilateralGrid ref_grid(num_images, grid_W, grid_H, grid_L);

    EXPECT_EQ(new_grid.grid_width(), grid_W);
    EXPECT_EQ(new_grid.grid_height(), grid_H);
    EXPECT_EQ(new_grid.grid_guidance(), grid_L);
    EXPECT_EQ(new_grid.num_images(), num_images);

    // Check parameter shapes
    auto new_params = new_grid.parameters();
    auto ref_params = ref_grid.parameters();

    EXPECT_EQ(new_params.shape()[0], num_images);
    EXPECT_EQ(new_params.shape()[1], 12);
    EXPECT_EQ(new_params.shape()[2], grid_L);
    EXPECT_EQ(new_params.shape()[3], grid_H);
    EXPECT_EQ(new_params.shape()[4], grid_W);

    EXPECT_EQ(ref_params.size(0), num_images);
    EXPECT_EQ(ref_params.size(1), 12);
}

// ===================================================================================
// Forward Pass Correctness Tests
// ===================================================================================

TEST_F(BilateralGridTest, ForwardPass_SmallImage) {
    const int H = 64, W = 64;
    const int num_images = 1;

    lfs::training::BilateralGrid new_grid(num_images);
    gs::training::BilateralGrid ref_grid(num_images);
    copy_grids(new_grid, ref_grid);

    // Create test image [H, W, 3] for new, [3, H, W] for ref
    auto test_image_hwc = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
    auto test_image_chw = to_torch(test_image_hwc).permute({2, 0, 1}).contiguous();

    // Forward pass
    auto [new_output, new_ctx] = new_grid.apply_forward(test_image_hwc, 0);
    auto [ref_output_chw, ref_ctx] = ref_grid.apply_forward(test_image_chw, 0);

    // Compare outputs (convert ref from [3, H, W] to [H, W, 3])
    auto ref_output_hwc = ref_output_chw.permute({1, 2, 0}).contiguous();

    EXPECT_TRUE(tensors_close(new_output, ref_output_hwc, 1e-4f, 1e-5f))
        << "Forward pass outputs don't match for " << H << "x" << W << " image";
}

TEST_F(BilateralGridTest, ForwardPass_MediumImage) {
    const int H = 256, W = 256;
    const int num_images = 5;

    lfs::training::BilateralGrid new_grid(num_images);
    gs::training::BilateralGrid ref_grid(num_images);
    copy_grids(new_grid, ref_grid);

    // Test multiple images
    for (int img_idx = 0; img_idx < num_images; img_idx++) {
        auto test_image_hwc = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
        auto test_image_chw = to_torch(test_image_hwc).permute({2, 0, 1}).contiguous();

        auto [new_output, new_ctx] = new_grid.apply_forward(test_image_hwc, img_idx);
        auto [ref_output_chw, ref_ctx] = ref_grid.apply_forward(test_image_chw, img_idx);

        auto ref_output_hwc = ref_output_chw.permute({1, 2, 0}).contiguous();

        EXPECT_TRUE(tensors_close(new_output, ref_output_hwc, 1e-4f, 1e-5f))
            << "Forward pass mismatch for image " << img_idx;
    }
}

TEST_F(BilateralGridTest, ForwardPass_LargeImage) {
    const int H = 800, W = 800;
    const int num_images = 1;

    lfs::training::BilateralGrid new_grid(num_images);
    gs::training::BilateralGrid ref_grid(num_images);
    copy_grids(new_grid, ref_grid);

    auto test_image_hwc = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
    auto test_image_chw = to_torch(test_image_hwc).permute({2, 0, 1}).contiguous();

    auto [new_output, new_ctx] = new_grid.apply_forward(test_image_hwc, 0);
    auto [ref_output_chw, ref_ctx] = ref_grid.apply_forward(test_image_chw, 0);

    auto ref_output_hwc = ref_output_chw.permute({1, 2, 0}).contiguous();

    EXPECT_TRUE(tensors_close(new_output, ref_output_hwc, 1e-4f, 1e-5f))
        << "Forward pass outputs don't match for " << H << "x" << W << " image";
}

// ===================================================================================
// Backward Pass Correctness Tests
// ===================================================================================

TEST_F(BilateralGridTest, BackwardPass_GradientAccumulation) {
    const int H = 128, W = 128;
    const int num_images = 3;

    lfs::training::BilateralGrid new_grid(num_images);
    gs::training::BilateralGrid ref_grid(num_images);
    copy_grids(new_grid, ref_grid);

    // Zero gradients
    new_grid.zero_grad();
    // Initialize gradient for reference implementation
    auto ref_params = ref_grid.parameters();
    if (!ref_params.grad().defined()) {
        ref_params.mutable_grad() = torch::zeros_like(ref_params);
    } else {
        ref_params.mutable_grad().zero_();
    }

    // Process multiple images and accumulate gradients
    for (int img_idx = 0; img_idx < num_images; img_idx++) {
        auto test_image_hwc = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
        auto test_image_chw = to_torch(test_image_hwc).permute({2, 0, 1}).contiguous();

        // Forward
        auto [new_output, new_ctx] = new_grid.apply_forward(test_image_hwc, img_idx);
        auto [ref_output_chw, ref_ctx] = ref_grid.apply_forward(test_image_chw, img_idx);

        // Create gradient [H, W, 3] for new, [3, H, W] for ref
        auto grad_output_hwc = lfs::core::Tensor::randn({H, W, 3}, lfs::core::Device::CUDA);
        auto grad_output_chw = to_torch(grad_output_hwc).permute({2, 0, 1}).contiguous();

        // Backward
        auto new_grad_rgb = new_grid.apply_backward(new_ctx, grad_output_hwc);
        auto ref_grad_rgb = ref_grid.apply_backward(ref_ctx, grad_output_chw, img_idx);

        // Check grad_rgb matches
        auto ref_grad_rgb_hwc = ref_grad_rgb.permute({1, 2, 0}).contiguous();
        EXPECT_TRUE(tensors_close(new_grad_rgb, ref_grad_rgb_hwc, 1e-4f, 1e-5f))
            << "grad_rgb mismatch for image " << img_idx;
    }

    // After all backward passes, check accumulated grid gradients
    auto new_grid_grad = new_grid.grad();
    auto ref_grid_grad = ref_grid.parameters().grad();

    EXPECT_TRUE(tensors_close(new_grid_grad, ref_grid_grad, 1e-4f, 1e-5f))
        << "Accumulated grid gradients don't match";
}

// ===================================================================================
// TV Loss Tests
// ===================================================================================

TEST_F(BilateralGridTest, TVLoss_Forward) {
    const int num_images = 10;

    lfs::training::BilateralGrid new_grid(num_images);
    gs::training::BilateralGrid ref_grid(num_images);
    copy_grids(new_grid, ref_grid);

    // Compute TV loss
    auto [new_tv_loss, new_tv_ctx] = new_grid.tv_loss_forward();
    auto [ref_tv_loss, ref_tv_ctx] = ref_grid.tv_loss_forward();

    EXPECT_TRUE(float_close(new_tv_loss, ref_tv_loss.item<float>(), 1e-4f, 1e-5f))
        << "TV loss mismatch: new=" << new_tv_loss << " vs ref=" << ref_tv_loss;
}

TEST_F(BilateralGridTest, TVLoss_Backward) {
    const int num_images = 10;

    lfs::training::BilateralGrid new_grid(num_images);
    gs::training::BilateralGrid ref_grid(num_images);
    copy_grids(new_grid, ref_grid);

    // Zero gradients
    new_grid.zero_grad();
    // Initialize gradient for reference implementation
    auto ref_params = ref_grid.parameters();
    if (!ref_params.grad().defined()) {
        ref_params.mutable_grad() = torch::zeros_like(ref_params);
    } else {
        ref_params.mutable_grad().zero_();
    }

    // Forward
    auto [new_tv_loss, new_tv_ctx] = new_grid.tv_loss_forward();
    auto [ref_tv_loss, ref_tv_ctx] = ref_grid.tv_loss_forward();

    // Backward with same grad_loss
    float grad_loss = 0.01f;
    new_grid.tv_loss_backward(new_tv_ctx, grad_loss);
    ref_grid.tv_loss_backward(ref_tv_ctx, grad_loss);

    // Check gradients
    auto new_grid_grad = new_grid.grad();
    auto ref_grid_grad = ref_grid.parameters().grad();

    EXPECT_TRUE(tensors_close(new_grid_grad, ref_grid_grad, 1e-4f, 1e-5f))
        << "TV loss gradients don't match";
}

// ===================================================================================
// Benchmark Tests - Realistic Workloads
// ===================================================================================

TEST_F(BilateralGridTest, DISABLED_Benchmark_Forward_SmallDataset) {
    // Small dataset: 100 images at 256x256
    const int num_images = 100;
    const int H = 256, W = 256;
    const int num_iterations = 100;

    lfs::training::BilateralGrid new_grid(num_images);
    gs::training::BilateralGrid ref_grid(num_images);
    copy_grids(new_grid, ref_grid);

    // Warmup
    auto warmup_image = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
    for (int i = 0; i < 10; i++) {
        auto [out, ctx] = new_grid.apply_forward(warmup_image, i % num_images);
    }
    cudaDeviceSynchronize();

    // Benchmark new implementation
    auto start_new = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto test_image = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
        auto [output, ctx] = new_grid.apply_forward(test_image, i % num_images);
    }
    cudaDeviceSynchronize();
    auto end_new = std::chrono::high_resolution_clock::now();

    // Benchmark reference implementation
    auto start_ref = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto test_image = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
        auto test_image_chw = to_torch(test_image).permute({2, 0, 1}).contiguous();
        auto [output, ctx] = ref_grid.apply_forward(test_image_chw, i % num_images);
    }
    cudaDeviceSynchronize();
    auto end_ref = std::chrono::high_resolution_clock::now();

    auto duration_new = std::chrono::duration_cast<std::chrono::milliseconds>(end_new - start_new).count();
    auto duration_ref = std::chrono::duration_cast<std::chrono::milliseconds>(end_ref - start_ref).count();

    std::cout << "\n=== Forward Pass Benchmark (100 images, 256x256, " << num_iterations << " iterations) ===" << std::endl;
    std::cout << "New implementation: " << duration_new << " ms" << std::endl;
    std::cout << "Ref implementation: " << duration_ref << " ms" << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2)
              << (static_cast<double>(duration_ref) / duration_new) << "x" << std::endl;
}

TEST_F(BilateralGridTest, DISABLED_Benchmark_Forward_LargeDataset_2K) {
    // Realistic: 2000 images at 800x800 (typical training scenario)
    const int num_images = 2000;
    const int H = 800, W = 800;
    const int num_iterations = 50;  // Process 50 random images

    std::cout << "\n=== Large Dataset Benchmark (2000 images, 800x800) ===" << std::endl;
    std::cout << "Initializing grids..." << std::flush;

    lfs::training::BilateralGrid new_grid(num_images);
    gs::training::BilateralGrid ref_grid(num_images);
    copy_grids(new_grid, ref_grid);

    std::cout << " done." << std::endl;
    std::cout << "Running warmup..." << std::flush;

    // Warmup
    auto warmup_image = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
    for (int i = 0; i < 5; i++) {
        auto [out, ctx] = new_grid.apply_forward(warmup_image, i % num_images);
    }
    cudaDeviceSynchronize();

    std::cout << " done." << std::endl;
    std::cout << "Benchmarking new implementation..." << std::flush;

    // Benchmark new implementation
    auto start_new = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto test_image = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
        int img_idx = rand() % num_images;
        auto [output, ctx] = new_grid.apply_forward(test_image, img_idx);
    }
    cudaDeviceSynchronize();
    auto end_new = std::chrono::high_resolution_clock::now();

    std::cout << " done." << std::endl;
    std::cout << "Benchmarking reference implementation..." << std::flush;

    // Benchmark reference implementation
    auto start_ref = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto test_image = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
        auto test_image_chw = to_torch(test_image).permute({2, 0, 1}).contiguous();
        int img_idx = rand() % num_images;
        auto [output, ctx] = ref_grid.apply_forward(test_image_chw, img_idx);
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

TEST_F(BilateralGridTest, DISABLED_Benchmark_Backward_FullPipeline_2K) {
    // Full training-like benchmark: forward + backward with 2000 images
    const int num_images = 2000;
    const int H = 800, W = 800;
    const int num_iterations = 50;

    std::cout << "\n=== Full Pipeline Benchmark (Forward + Backward, 2000 images, 800x800) ===" << std::endl;
    std::cout << "Initializing grids..." << std::flush;

    lfs::training::BilateralGrid new_grid(num_images);
    gs::training::BilateralGrid ref_grid(num_images);
    copy_grids(new_grid, ref_grid);

    std::cout << " done." << std::endl;

    // Benchmark new implementation (forward + backward)
    std::cout << "Benchmarking new implementation..." << std::flush;
    new_grid.zero_grad();
    auto start_new = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto test_image = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
        int img_idx = rand() % num_images;

        // Forward
        auto [output, ctx] = new_grid.apply_forward(test_image, img_idx);

        // Backward
        auto grad_output = lfs::core::Tensor::randn({H, W, 3}, lfs::core::Device::CUDA);
        auto grad_rgb = new_grid.apply_backward(ctx, grad_output);
    }
    cudaDeviceSynchronize();
    auto end_new = std::chrono::high_resolution_clock::now();

    std::cout << " done." << std::endl;

    // Benchmark reference implementation (forward + backward)
    std::cout << "Benchmarking reference implementation..." << std::flush;
    auto ref_params = ref_grid.parameters();
    if (!ref_params.grad().defined()) {
        ref_params.mutable_grad() = torch::zeros_like(ref_params);
    } else {
        ref_params.mutable_grad().zero_();
    }
    auto start_ref = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto test_image = lfs::core::Tensor::rand({H, W, 3}, lfs::core::Device::CUDA);
        auto test_image_chw = to_torch(test_image).permute({2, 0, 1}).contiguous();
        int img_idx = rand() % num_images;

        // Forward
        auto [output, ctx] = ref_grid.apply_forward(test_image_chw, img_idx);

        // Backward
        auto grad_output_hwc = lfs::core::Tensor::randn({H, W, 3}, lfs::core::Device::CUDA);
        auto grad_output_chw = to_torch(grad_output_hwc).permute({2, 0, 1}).contiguous();
        auto grad_rgb = ref_grid.apply_backward(ctx, grad_output_chw, img_idx);
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
