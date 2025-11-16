/* Test to isolate tensor struct initialization bug */

#include "core_new/logger.hpp"
#include "core_new/tensor.hpp"
#include <gtest/gtest.h>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

class TensorStructInitTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup if needed
    }
};

TEST_F(TensorStructInitTest, DirectInitialization) {
    size_t N = 10;
    Device device = Device::CUDA;

    Tensor t1 = Tensor::empty({N, 3}, device);
    Tensor t2 = Tensor::ones({5, 3}, device);

    // Copy data
    t1.slice(0, 0, 5).copy_(t2);

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, DirectInitializationWithDtype) {
    size_t N = 10;
    Device device = Device::CUDA;
    DataType dtype = DataType::Float32;

    Tensor t1 = Tensor::empty({N, 3}, device, dtype);
    Tensor t2 = Tensor::ones({5, 3}, device);

    // Copy data
    t1.slice(0, 0, 5).copy_(t2);

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, StructAggregateInitializationWithDtype) {
    size_t N = 10;
    Device device = Device::CUDA;
    DataType dtype = DataType::Float32;

    struct TensorPair {
        Tensor t1, t2;
    } tensors{
        .t1 = Tensor::empty({N, 3}, device, dtype),
        .t2 = Tensor::empty({N, 3}, device, dtype)};

    Tensor source = Tensor::ones({5, 3}, device);

    // Copy data
    tensors.t1.slice(0, 0, 5).copy_(source);

    auto cpu = tensors.t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Struct aggregate initialization with dtype failed";
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, StructAggregateInitializationWithoutDtype) {
    size_t N = 10;
    Device device = Device::CUDA;

    struct TensorPair {
        Tensor t1, t2;
    } tensors{
        .t1 = Tensor::empty({N, 3}, device),
        .t2 = Tensor::empty({N, 3}, device)};

    Tensor source = Tensor::ones({5, 3}, device);

    // Copy data
    tensors.t1.slice(0, 0, 5).copy_(source);

    auto cpu = tensors.t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Struct aggregate initialization without dtype failed";
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, CopyFromVsCopyUnderscore) {
    size_t N = 10;
    Device device = Device::CUDA;

    // Test copy_from
    {
        Tensor t1 = Tensor::empty({N, 3}, device);
        Tensor t2 = Tensor::ones({5, 3}, device);

        t1.slice(0, 0, 5).copy_from(t2);

        auto cpu = t1.cpu();
        auto ptr = cpu.ptr<float>();

        EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "copy_from failed";
        EXPECT_FLOAT_EQ(ptr[1], 1.0f);
        EXPECT_FLOAT_EQ(ptr[2], 1.0f);
    }

    // Test copy_
    {
        Tensor t1 = Tensor::empty({N, 3}, device);
        Tensor t2 = Tensor::ones({5, 3}, device);

        t1.slice(0, 0, 5).copy_(t2);

        auto cpu = t1.cpu();
        auto ptr = cpu.ptr<float>();

        EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "copy_ failed";
        EXPECT_FLOAT_EQ(ptr[1], 1.0f);
        EXPECT_FLOAT_EQ(ptr[2], 1.0f);
    }
}

TEST_F(TensorStructInitTest, SliceAssignmentOperator) {
    size_t N = 10;
    Device device = Device::CUDA;

    Tensor t1 = Tensor::empty({N, 3}, device);
    Tensor t2 = Tensor::ones({5, 3}, device);

    // Use assignment operator (this might be broken)
    t1.slice(0, 0, 5) = t2;

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Slice assignment operator failed";
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, ComplexStructLikeOriginalScene) {
    size_t N = 100;
    Device device = Device::CUDA;
    DataType dtype = DataType::Float32;

    // Simulate the original broken pattern
    struct CombinedTensors {
        Tensor means, sh0, shN, opacity, scaling, rotation;
    } combined{
        .means = Tensor::empty({N, 3}, device, dtype),
        .sh0 = Tensor::empty({N, 1, 3}, device, dtype),
        .shN = Tensor::zeros({N, 15, 3}, device, dtype),
        .opacity = Tensor::empty({N, 1}, device, dtype),
        .scaling = Tensor::empty({N, 3}, device, dtype),
        .rotation = Tensor::empty({N, 4}, device, dtype)};

    // Create source data
    Tensor source_sh0 = Tensor::ones({10, 1, 3}, device);

    // Try to copy using assignment operator
    combined.sh0.slice(0, 0, 10) = source_sh0;

    auto cpu = combined.sh0.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Complex struct pattern with assignment operator failed";
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, ComplexStructWithCopy) {
    size_t N = 100;
    Device device = Device::CUDA;
    DataType dtype = DataType::Float32;

    // Simulate the fixed pattern
    struct CombinedTensors {
        Tensor means, sh0, shN, opacity, scaling, rotation;
    } combined{
        .means = Tensor::empty({N, 3}, device, dtype),
        .sh0 = Tensor::empty({N, 1, 3}, device, dtype),
        .shN = Tensor::zeros({N, 15, 3}, device, dtype),
        .opacity = Tensor::empty({N, 1}, device, dtype),
        .scaling = Tensor::empty({N, 3}, device, dtype),
        .rotation = Tensor::empty({N, 4}, device, dtype)};

    // Create source data
    Tensor source_sh0 = Tensor::ones({10, 1, 3}, device);

    // Try to copy using copy_
    combined.sh0.slice(0, 0, 10).copy_(source_sh0);

    auto cpu = combined.sh0.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Complex struct pattern with copy_ failed";
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

// ============================================================================
// COMPREHENSIVE TESTS TO EXPOSE ASSIGNMENT OPERATOR BUG
// ============================================================================

TEST_F(TensorStructInitTest, DirectAssignmentNoSlice) {
    Device device = Device::CUDA;

    Tensor t1 = Tensor::empty({5, 3}, device);
    Tensor t2 = Tensor::ones({5, 3}, device);

    // Direct assignment without slicing
    t1 = t2;

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Direct assignment (no slice) failed";
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, SliceAssignmentSameSize) {
    Device device = Device::CUDA;

    Tensor t1 = Tensor::empty({10, 3}, device);
    Tensor t2 = Tensor::ones({5, 3}, device);

    // Slice assignment - same size as source
    t1.slice(0, 0, 5) = t2;

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Slice assignment (same size) failed";
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, SliceAssignmentOffset) {
    Device device = Device::CUDA;

    Tensor t1 = Tensor::zeros({10, 3}, device);
    Tensor t2 = Tensor::ones({3, 3}, device) * 2.0f;

    // Slice assignment - with offset
    t1.slice(0, 5, 8) = t2;

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    // Check that offset region was written
    EXPECT_FLOAT_EQ(ptr[5 * 3 + 0], 2.0f) << "Slice assignment at offset failed";
    EXPECT_FLOAT_EQ(ptr[5 * 3 + 1], 2.0f);
    EXPECT_FLOAT_EQ(ptr[5 * 3 + 2], 2.0f);

    // Check that beginning is still zero
    EXPECT_FLOAT_EQ(ptr[0], 0.0f) << "Slice assignment corrupted data before offset";
}

TEST_F(TensorStructInitTest, SliceAssignment3D) {
    Device device = Device::CUDA;

    Tensor t1 = Tensor::empty({10, 1, 3}, device);
    Tensor t2 = Tensor::ones({5, 1, 3}, device);

    // 3D tensor slice assignment
    t1.slice(0, 0, 5) = t2;

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "3D slice assignment failed";
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, SliceAssignmentLargeTensor) {
    Device device = Device::CUDA;

    Tensor t1 = Tensor::empty({1000000, 3}, device);
    Tensor t2 = Tensor::ones({100000, 3}, device);

    // Large tensor slice assignment
    t1.slice(0, 0, 100000) = t2;

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Large tensor slice assignment failed at start";
    EXPECT_FLOAT_EQ(ptr[99999 * 3 + 0], 1.0f) << "Large tensor slice assignment failed at end";
}

TEST_F(TensorStructInitTest, MultipleSliceAssignments) {
    Device device = Device::CUDA;

    Tensor t1 = Tensor::zeros({100, 3}, device);
    Tensor t2 = Tensor::ones({10, 3}, device);
    Tensor t3 = Tensor::ones({10, 3}, device) * 2.0f;
    Tensor t4 = Tensor::ones({10, 3}, device) * 3.0f;

    // Multiple slice assignments
    t1.slice(0, 0, 10) = t2;
    t1.slice(0, 10, 20) = t3;
    t1.slice(0, 20, 30) = t4;

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0 * 3 + 0], 1.0f) << "First slice assignment failed";
    EXPECT_FLOAT_EQ(ptr[10 * 3 + 0], 2.0f) << "Second slice assignment failed";
    EXPECT_FLOAT_EQ(ptr[20 * 3 + 0], 3.0f) << "Third slice assignment failed";
    EXPECT_FLOAT_EQ(ptr[30 * 3 + 0], 0.0f) << "Region after assignments should still be zero";
}

TEST_F(TensorStructInitTest, SliceAssignmentFromView) {
    Device device = Device::CUDA;

    Tensor t1 = Tensor::empty({10, 3}, device);
    Tensor t2 = Tensor::ones({20, 3}, device);

    // Assign from a view/slice of another tensor
    t1.slice(0, 0, 5) = t2.slice(0, 5, 10);

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Slice assignment from view failed";
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, SliceAssignmentChained) {
    Device device = Device::CUDA;

    Tensor t1 = Tensor::empty({10, 5, 3}, device);
    Tensor t2 = Tensor::ones({5, 5, 3}, device);

    // Chained slice assignment (slice in first dimension only)
    t1.slice(0, 0, 5) = t2;

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Chained slice assignment failed";
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, SliceAssignmentMultiDimSlice) {
    Device device = Device::CUDA;

    Tensor t1 = Tensor::zeros({10, 15, 3}, device);
    Tensor t2 = Tensor::ones({5, 8, 3}, device);

    // Multi-dimensional slicing
    auto dst_slice = t1.slice(0, 0, 5).slice(1, 0, 8);
    dst_slice = t2;

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Multi-dimensional slice assignment failed";
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
}

TEST_F(TensorStructInitTest, CompareAssignmentVsCopyMethods) {
    Device device = Device::CUDA;

    // Test 1: Assignment operator (should NOW work after fix!)
    {
        Tensor t1 = Tensor::zeros({10, 3}, device);
        Tensor t2 = Tensor::ones({5, 3}, device);
        t1.slice(0, 0, 5) = t2;

        auto cpu = t1.cpu();
        auto ptr = cpu.ptr<float>();

        EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "Assignment operator should work after fix";
        EXPECT_FLOAT_EQ(ptr[1], 1.0f);
        EXPECT_FLOAT_EQ(ptr[2], 1.0f);
    }

    // Test 2: copy_ method
    {
        Tensor t1 = Tensor::zeros({10, 3}, device);
        Tensor t2 = Tensor::ones({5, 3}, device);
        t1.slice(0, 0, 5).copy_(t2);

        auto cpu = t1.cpu();
        auto ptr = cpu.ptr<float>();

        EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "copy_ method should work";
    }

    // Test 3: copy_from method
    {
        Tensor t1 = Tensor::zeros({10, 3}, device);
        Tensor t2 = Tensor::ones({5, 3}, device);
        t1.slice(0, 0, 5).copy_from(t2);

        auto cpu = t1.cpu();
        auto ptr = cpu.ptr<float>();

        EXPECT_FLOAT_EQ(ptr[0], 1.0f) << "copy_from method should work";
    }
}

TEST_F(TensorStructInitTest, SliceAssignmentPreservesOtherData) {
    Device device = Device::CUDA;

    // Fill tensor with known values
    Tensor t1 = Tensor::ones({10, 3}, device) * 5.0f;
    Tensor t2 = Tensor::ones({3, 3}, device) * 7.0f;

    // Assign to middle slice
    t1.slice(0, 3, 6) = t2;

    auto cpu = t1.cpu();
    auto ptr = cpu.ptr<float>();

    // Check that data before slice is preserved (should be 5.0)
    EXPECT_FLOAT_EQ(ptr[0 * 3 + 0], 5.0f) << "Data before slice should be preserved";
    EXPECT_FLOAT_EQ(ptr[2 * 3 + 0], 5.0f) << "Data before slice should be preserved";

    // Check that sliced region has new value (should be 7.0 if working, 0.0 if bug)
    // This documents the bug - assignment zeros the data
    float sliced_value = ptr[3 * 3 + 0];
    if (sliced_value == 0.0f) {
        FAIL() << "CONFIRMED BUG: Slice assignment zeros data instead of copying (expected 7.0, got " << sliced_value << ")";
    } else if (sliced_value == 7.0f) {
        SUCCEED() << "Slice assignment works correctly";
    } else {
        FAIL() << "Unexpected value in sliced region: " << sliced_value;
    }

    // Check that data after slice is preserved (should be 5.0)
    EXPECT_FLOAT_EQ(ptr[7 * 3 + 0], 5.0f) << "Data after slice should be preserved";
    EXPECT_FLOAT_EQ(ptr[9 * 3 + 0], 5.0f) << "Data after slice should be preserved";
}
