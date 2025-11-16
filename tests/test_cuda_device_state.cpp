/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Test to check if LibTorch changes CUDA device state

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace lfs::core;

void print_cuda_device_state() {
    int device;
    cudaGetDevice(&device);
    std::cout << "  Current CUDA device: " << device << std::endl;

    int count;
    cudaGetDeviceCount(&count);
    std::cout << "  Total CUDA devices: " << count << std::endl;
}

TEST(CUDADeviceState, InitialState) {
    std::cout << "Initial CUDA state:" << std::endl;
    print_cuda_device_state();
    EXPECT_EQ(0, 0); // Just to make test pass
}

TEST(CUDADeviceState, AfterLFSTensorOp) {
    std::cout << "CUDA state before LFS tensor op:" << std::endl;
    print_cuda_device_state();

    auto t = Tensor::randn({100000, 4}, Device::CUDA);
    std::vector<int> idx_data(50000);
    for (int i = 0; i < 50000; ++i) idx_data[i] = i;
    auto indices = Tensor::from_vector(idx_data, TensorShape({50000}), Device::CUDA);
    auto result = t.index_select(0, indices);

    std::cout << "CUDA state after LFS tensor op:" << std::endl;
    print_cuda_device_state();
}

TEST(CUDADeviceState, AfterTorchOp) {
    std::cout << "CUDA state before Torch op:" << std::endl;
    print_cuda_device_state();

    auto t = torch::randn({100000, 4}, torch::kCUDA);
    auto idx = torch::arange(0, 50000, torch::kLong).to(torch::kCUDA);
    auto result = t.index_select(0, idx);

    std::cout << "CUDA state after Torch op:" << std::endl;
    print_cuda_device_state();
}

TEST(CUDADeviceState, TorchThenLFS) {
    std::cout << "CUDA state before Torch op:" << std::endl;
    print_cuda_device_state();

    // Run torch operation
    auto t_torch = torch::randn({100000, 4}, torch::kCUDA);
    auto idx_torch = torch::arange(0, 50000, torch::kLong).to(torch::kCUDA);
    auto result_torch = t_torch.index_select(0, idx_torch);

    std::cout << "CUDA state after Torch op:" << std::endl;
    print_cuda_device_state();

    // Now run LFS operation
    try {
        auto t_lfs = Tensor::randn({100000, 4}, Device::CUDA);
        std::vector<int> idx_data_lfs(50000);
        for (int i = 0; i < 50000; ++i) idx_data_lfs[i] = i;
        auto idx_lfs = Tensor::from_vector(idx_data_lfs, TensorShape({50000}), Device::CUDA);
        auto result_lfs = t_lfs.index_select(0, idx_lfs);

        std::cout << "CUDA state after LFS op (SUCCESS):" << std::endl;
        print_cuda_device_state();
    } catch (const std::exception& e) {
        std::cout << "LFS op FAILED after Torch: " << e.what() << std::endl;
        std::cout << "CUDA state after LFS op (FAILED):" << std::endl;
        print_cuda_device_state();
        FAIL() << "LFS operation failed after Torch operation: " << e.what();
    }
}

TEST(CUDADeviceState, ResetDeviceBeforeLFS) {
    std::cout << "CUDA state initial:" << std::endl;
    print_cuda_device_state();

    // Run torch operation
    auto t_torch = torch::randn({100000, 4}, torch::kCUDA);
    auto idx_torch = torch::arange(0, 50000, torch::kLong).to(torch::kCUDA);
    auto result_torch = t_torch.index_select(0, idx_torch);

    std::cout << "CUDA state after Torch op:" << std::endl;
    print_cuda_device_state();

    // EXPLICITLY RESET TO DEVICE 0
    cudaSetDevice(0);
    std::cout << "CUDA state after cudaSetDevice(0):" << std::endl;
    print_cuda_device_state();

    // Now run LFS operation
    try {
        auto t_lfs = Tensor::randn({100000, 4}, Device::CUDA);
        std::vector<int> idx_data_lfs(50000);
        for (int i = 0; i < 50000; ++i) idx_data_lfs[i] = i;
        auto idx_lfs = Tensor::from_vector(idx_data_lfs, TensorShape({50000}), Device::CUDA);
        auto result_lfs = t_lfs.index_select(0, idx_lfs);

        std::cout << "LFS op SUCCESS after cudaSetDevice(0)" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "LFS op STILL FAILED even after cudaSetDevice(0): " << e.what() << std::endl;
        FAIL() << "LFS operation failed even after cudaSetDevice(0): " << e.what();
    }
}
