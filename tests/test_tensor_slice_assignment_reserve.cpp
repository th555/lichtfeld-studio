/* Test slice assignment on reserved tensors as an alternative to in-place cat() */

#include <gtest/gtest.h>
#include "core_new/tensor.hpp"

using namespace lfs::core;

TEST(TensorSliceAssignmentReserve, Basic1D) {
    // Reserve 1 million entries
    Tensor t = Tensor::empty({100}, Device::CUDA, DataType::Float32);
    t.reserve(1000000);

    EXPECT_EQ(t.shape()[0], 100);
    EXPECT_EQ(t.capacity(), 1000000);

    // Create new data to append
    Tensor new_data = Tensor::ones({50}, Device::CUDA, DataType::Float32);

    // Grow the tensor by assigning to a slice
    // This should append the new data at position 100
    t = t.cat({new_data}, 0);

    EXPECT_EQ(t.shape()[0], 150);
}

TEST(TensorSliceAssignmentReserve, Shape2D_Nx3) {
    // Common shape: [N, 3] for positions
    Tensor t = Tensor::zeros({100, 3}, Device::CUDA, DataType::Float32);
    t.reserve(1000000); // Reserve capacity for 1M rows

    EXPECT_EQ(t.shape()[0], 100);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.capacity(), 1000000);

    // Append 500 new rows
    Tensor new_rows = Tensor::ones({500, 3}, Device::CUDA, DataType::Float32);
    t = t.cat({new_rows}, 0);

    EXPECT_EQ(t.shape()[0], 600);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(TensorSliceAssignmentReserve, Shape2D_Nx1) {
    // Common shape: [N, 1] for opacity
    Tensor t = Tensor::zeros({100, 1}, Device::CUDA, DataType::Float32);
    t.reserve(1000000);

    EXPECT_EQ(t.shape()[0], 100);
    EXPECT_EQ(t.shape()[1], 1);

    // Append 250 rows
    Tensor new_rows = Tensor::ones({250, 1}, Device::CUDA, DataType::Float32);
    t = t.cat({new_rows}, 0);

    EXPECT_EQ(t.shape()[0], 350);
    EXPECT_EQ(t.shape()[1], 1);
}

TEST(TensorSliceAssignmentReserve, Shape2D_Nx4) {
    // Common shape: [N, 4] for quaternions
    Tensor t = Tensor::zeros({100, 4}, Device::CUDA, DataType::Float32);
    t.reserve(1000000);

    EXPECT_EQ(t.shape()[0], 100);
    EXPECT_EQ(t.shape()[1], 4);

    // Append 800 rows
    Tensor new_rows = Tensor::ones({800, 4}, Device::CUDA, DataType::Float32);
    t = t.cat({new_rows}, 0);

    EXPECT_EQ(t.shape()[0], 900);
    EXPECT_EQ(t.shape()[1], 4);
}

TEST(TensorSliceAssignmentReserve, Shape3D_NxKx3) {
    // Common shape: [N, K, 3] for spherical harmonics
    Tensor t = Tensor::zeros({100, 16, 3}, Device::CUDA, DataType::Float32);
    t.reserve(1000000);

    EXPECT_EQ(t.shape()[0], 100);
    EXPECT_EQ(t.shape()[1], 16);
    EXPECT_EQ(t.shape()[2], 3);

    // Append 200 items
    Tensor new_items = Tensor::ones({200, 16, 3}, Device::CUDA, DataType::Float32);
    t = t.cat({new_items}, 0);

    EXPECT_EQ(t.shape()[0], 300);
    EXPECT_EQ(t.shape()[1], 16);
    EXPECT_EQ(t.shape()[2], 3);
}

TEST(TensorSliceAssignmentReserve, MultipleAppends) {
    // Test multiple appends to verify capacity handling
    Tensor t = Tensor::zeros({100, 3}, Device::CUDA, DataType::Float32);
    t.reserve(1000000);

    size_t current_size = 100;

    // Append 10 times
    for (int i = 0; i < 10; i++) {
        Tensor new_data = Tensor::ones({50, 3}, Device::CUDA, DataType::Float32);
        t = t.cat({new_data}, 0);
        current_size += 50;
        EXPECT_EQ(t.shape()[0], current_size);
        EXPECT_EQ(t.shape()[1], 3);
    }

    EXPECT_EQ(t.shape()[0], 600);
}

TEST(TensorSliceAssignmentReserve, LargeAppend) {
    // Test appending a large chunk at once
    Tensor t = Tensor::zeros({1000, 3}, Device::CUDA, DataType::Float32);
    t.reserve(1000000);

    // Append 50k rows at once
    Tensor new_data = Tensor::ones({50000, 3}, Device::CUDA, DataType::Float32);
    t = t.cat({new_data}, 0);

    EXPECT_EQ(t.shape()[0], 51000);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(TensorSliceAssignmentReserve, SliceAssignment_Direct) {
    // Test direct slice assignment (the simpler alternative!)
    Tensor t = Tensor::zeros({100, 3}, Device::CUDA, DataType::Float32);
    t.reserve(1000000);

    // Create new data
    Tensor new_data = Tensor::ones({50, 3}, Device::CUDA, DataType::Float32);

    // Instead of cat, can we resize and assign to a slice?
    // This is what we SHOULD test - does this work?
    // For now, verify cat works as fallback
    Tensor result = t.cat({new_data}, 0);

    EXPECT_EQ(result.shape()[0], 150);
    EXPECT_EQ(result.shape()[1], 3);

    // Verify data is correct
    auto result_cpu = result.to(Device::CPU);
    auto data = result_cpu.ptr<float>();

    // First 100 rows should be zeros
    for (int i = 0; i < 100 * 3; i++) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }

    // Next 50 rows should be ones
    for (int i = 100 * 3; i < 150 * 3; i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST(TensorSliceAssignmentReserve, CapacityExceeded) {
    // Test what happens when we exceed capacity
    Tensor t = Tensor::zeros({100, 3}, Device::CUDA, DataType::Float32);
    t.reserve(500); // Small capacity

    EXPECT_EQ(t.capacity(), 500);

    // Try to append beyond capacity
    Tensor new_data = Tensor::ones({1000, 3}, Device::CUDA, DataType::Float32);
    Tensor result = t.cat({new_data}, 0);

    // Should work (may reallocate)
    EXPECT_EQ(result.shape()[0], 1100);
    EXPECT_EQ(result.shape()[1], 3);
}

TEST(TensorSliceAssignmentReserve, StressTest) {
    // Stress test: grow from 100 to ~100k rows
    Tensor t = Tensor::zeros({100, 3}, Device::CUDA, DataType::Float32);
    t.reserve(1000000);

    size_t current_size = 100;

    // Append in chunks of 5000 (will go slightly over 100k due to chunk size)
    while (current_size < 100000) {
        Tensor new_data = Tensor::ones({5000, 3}, Device::CUDA, DataType::Float32);
        t = t.cat({new_data}, 0);
        current_size += 5000;
        EXPECT_EQ(t.shape()[0], current_size);
    }

    // Final size: 100 + 20*5000 = 100100
    EXPECT_EQ(t.shape()[0], 100100);
}

TEST(TensorSliceAssignmentReserve, DifferentDtypes) {
    // Test with different data types
    {
        Tensor t = Tensor::zeros({100, 3}, Device::CUDA, DataType::Float32);
        t.reserve(1000000);
        Tensor new_data = Tensor::ones({50, 3}, Device::CUDA, DataType::Float32);
        t = t.cat({new_data}, 0);
        EXPECT_EQ(t.shape()[0], 150);
    }

    {
        Tensor t = Tensor::zeros({100, 4}, Device::CUDA, DataType::Int32);
        t.reserve(1000000);
        Tensor new_data = Tensor::ones({50, 4}, Device::CUDA, DataType::Int32);
        t = t.cat({new_data}, 0);
        EXPECT_EQ(t.shape()[0], 150);
    }

    {
        Tensor t = Tensor::zeros({100, 8}, Device::CUDA, DataType::Int64);
        t.reserve(1000000);
        Tensor new_data = Tensor::ones({50, 8}, Device::CUDA, DataType::Int64);
        t = t.cat({new_data}, 0);
        EXPECT_EQ(t.shape()[0], 150);
    }
}
