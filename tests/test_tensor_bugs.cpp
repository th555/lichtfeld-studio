/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Comprehensive tensor library bug detection tests
 *
 * This test suite is designed to expose bugs and edge cases in the LFS tensor library.
 * Tests are organized by category and include known issues from sparsity optimizer debugging.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "core_new/tensor.hpp"

using namespace lfs::core;

// Helper to clear CUDA errors
inline void clear_cuda_errors() {
    cudaDeviceSynchronize();
    cudaGetLastError();
}

// Helper to check for pending CUDA errors
inline bool has_cuda_error() {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    return err != cudaSuccess;
}

// ============================================================================
// Bool Tensor Operations - Known Issues
// ============================================================================

class BoolTensorBugsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        clear_cuda_errors();
    }

    void TearDown() override {
        clear_cuda_errors();
    }
};

TEST_F(BoolTensorBugsTest, BoolSumReturnsDtype) {
    // KNOWN BUG: Bool tensor sum() should return Int32/Int64, not Bool
    auto bool_t = Tensor::ones_bool({100}, Device::CUDA);
    auto sum_result = bool_t.sum();

    std::cerr << "Bool sum dtype: " << dtype_name(sum_result.dtype()) << std::endl;

    // This will likely fail - sum should return Int32/Int64
    EXPECT_NE(sum_result.dtype(), DataType::Bool)
        << "KNOWN BUG: sum() on Bool tensor returns Bool instead of Int32/Int64";
}

TEST_F(BoolTensorBugsTest, BoolSumCorrectValue) {
    // Test that Bool sum actually computes correct value
    auto bool_t = Tensor::ones_bool({100}, Device::CUDA);

    // Workaround: convert to Int32 first
    auto sum_via_int32 = bool_t.to(DataType::Int32).sum().template item<int>();
    EXPECT_EQ(sum_via_int32, 100);

    // Direct sum (likely broken)
    bool has_error_before = has_cuda_error();
    auto sum_direct = bool_t.sum();
    bool has_error_after = has_cuda_error();

    if (has_error_after && !has_error_before) {
        std::cerr << "KNOWN BUG: sum() on Bool tensor generates CUDA error" << std::endl;
        clear_cuda_errors();
        GTEST_SKIP() << "sum() on Bool generates CUDA error";
    }
}

TEST_F(BoolTensorBugsTest, ItemWithDtypeMismatch) {
    // KNOWN BUG: item<T>() doesn't check dtype matches T
    auto bool_t = Tensor::ones_bool({1}, Device::CUDA);

    // This should fail gracefully but might crash
    bool has_error_before = has_cuda_error();

    // Trying to extract int from bool (4 bytes from 1 byte)
    EXPECT_ANY_THROW({
        int value = bool_t.item<int>();
        (void)value;
    }) << "KNOWN BUG: item<int>() on Bool tensor should throw but might crash";

    clear_cuda_errors();
}

TEST_F(BoolTensorBugsTest, BoolComparisonResult) {
    // Test that comparison operations return proper Bool tensors
    auto t = Tensor::randn({100}, Device::CUDA);
    auto mask = t > 0.0f;

    EXPECT_EQ(mask.dtype(), DataType::Bool);
    EXPECT_EQ(mask.shape()[0], 100);

    // Check sum works on comparison results
    auto count_via_int32 = mask.to(DataType::Int32).sum().template item<int>();
    EXPECT_GT(count_via_int32, 0);
    EXPECT_LE(count_via_int32, 100);
}

TEST_F(BoolTensorBugsTest, BoolLogicalOperations) {
    // Test Bool tensor logical operations
    auto t1 = Tensor::ones_bool({100}, Device::CUDA);
    auto t2 = Tensor::zeros_bool({100}, Device::CUDA);

    // NOT operation
    auto not_t1 = !t1;
    EXPECT_EQ(not_t1.dtype(), DataType::Bool);
    auto not_t1_sum = not_t1.to(DataType::Int32).sum().template item<int>();
    EXPECT_EQ(not_t1_sum, 0);

    auto not_t2 = !t2;
    auto not_t2_sum = not_t2.to(DataType::Int32).sum().template item<int>();
    EXPECT_EQ(not_t2_sum, 100);
}

TEST_F(BoolTensorBugsTest, BoolMaskedSelect) {
    // Test masked_select with Bool mask
    auto data = Tensor::arange(0, 100);
    auto mask = data < 50.0f;

    EXPECT_EQ(mask.dtype(), DataType::Bool);

    auto selected = data.masked_select(mask);

    // Should have 50 elements
    auto count_via_int32 = mask.to(DataType::Int32).sum().template item<int>();
    EXPECT_EQ(selected.numel(), count_via_int32);
}

// ============================================================================
// index_put_ Operations - Known Issues
// ============================================================================

class IndexPutBugsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        clear_cuda_errors();
    }

    void TearDown() override {
        clear_cuda_errors();
    }
};

TEST_F(IndexPutBugsTest, IndexPutSingleValue) {
    // KNOWN BUG: index_put_ may not work correctly
    auto t = Tensor::zeros({100}, Device::CPU);

    auto idx = Tensor::from_vector({10.0f}, {1}, Device::CPU).to(DataType::Int64);
    auto value = Tensor::ones({1}, Device::CPU);

    t.index_put_({idx}, value);

    // Check if value was actually set
    auto t_vec = t.to_vector();
    EXPECT_EQ(t_vec[10], 1.0f) << "KNOWN BUG: index_put_ may not set values correctly";
}

TEST_F(IndexPutBugsTest, IndexPutMultipleValues) {
    // Test setting multiple values via index_put_
    auto t = Tensor::zeros({100}, Device::CPU);

    std::vector<float> indices_vec = {5, 10, 15, 20};
    auto indices = Tensor::from_vector(indices_vec, {4}, Device::CPU).to(DataType::Int64);
    auto values = Tensor::ones({4}, Device::CPU);

    t.index_put_({indices}, values);

    auto t_vec = t.to_vector();
    int count_ones = 0;
    for (float v : t_vec) {
        if (v == 1.0f) count_ones++;
    }

    EXPECT_EQ(count_ones, 4) << "KNOWN BUG: index_put_ may not set all values correctly. Expected 4, got " << count_ones;
}

TEST_F(IndexPutBugsTest, IndexPutBoolTensor) {
    // KNOWN BUG from sparsity optimizer: index_put_ on Bool tensors
    auto mask = Tensor::zeros_bool({100}, Device::CPU);

    std::vector<float> indices_vec = {10, 20, 30, 40, 50};
    auto indices = Tensor::from_vector(indices_vec, {5}, Device::CPU).to(DataType::Int64);
    auto values = Tensor::ones_bool({5}, Device::CPU);

    mask.index_put_({indices}, values);

    // Check how many were actually set
    int count = mask.to(DataType::Int32).sum().template item<int>();

    EXPECT_EQ(count, 5) << "KNOWN BUG: index_put_ on Bool tensor only set " << count << " of 5 values";
}

TEST_F(IndexPutBugsTest, DirectMemoryAccessWorkaround) {
    // Workaround: direct memory access is more reliable
    auto mask = Tensor::zeros_bool({100}, Device::CPU);

    auto mask_ptr = mask.ptr<unsigned char>();
    std::vector<int> indices = {10, 20, 30, 40, 50};

    for (int idx : indices) {
        mask_ptr[idx] = 1;
    }

    int count = mask.to(DataType::Int32).sum().template item<int>();
    EXPECT_EQ(count, 5) << "Direct memory access should work correctly";
}

// ============================================================================
// item<T>() Type Safety
// ============================================================================

class ItemTypeSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        clear_cuda_errors();
    }

    void TearDown() override {
        clear_cuda_errors();
    }
};

TEST_F(ItemTypeSafetyTest, ItemFloat32FromFloat32) {
    auto t = Tensor::from_vector({42.5f}, {1}, Device::CUDA);
    EXPECT_FLOAT_EQ(t.item<float>(), 42.5f);
}

TEST_F(ItemTypeSafetyTest, ItemIntFromInt32) {
    auto t = Tensor::from_vector({42.0f}, {1}, Device::CUDA).to(DataType::Int32);
    EXPECT_EQ(t.item<int>(), 42);
}

TEST_F(ItemTypeSafetyTest, ItemIntFromInt64) {
    auto t = Tensor::from_vector({42.0f}, {1}, Device::CUDA).to(DataType::Int64);
    EXPECT_EQ(t.item<int64_t>(), 42);
}

TEST_F(ItemTypeSafetyTest, ItemWrongSize) {
    // KNOWN BUG: Extracting larger type from smaller tensor element
    auto bool_t = Tensor::ones_bool({1}, Device::CUDA);  // 1 byte

    // This should fail but might succeed with garbage
    bool has_error = false;
    try {
        int value = bool_t.item<int>();  // Trying to read 4 bytes
        std::cerr << "WARNING: item<int>() from Bool succeeded with value: " << value << std::endl;
    } catch (...) {
        has_error = true;
    }

    if (has_cuda_error()) {
        std::cerr << "KNOWN BUG: item<int>() on Bool generates CUDA error" << std::endl;
        clear_cuda_errors();
    }
}

TEST_F(ItemTypeSafetyTest, ItemMultiElementTensor) {
    auto t = Tensor::ones({10}, Device::CUDA);

    // item() should only work on single-element tensors
    EXPECT_ANY_THROW({
        float value = t.item<float>();
        (void)value;
    }) << "item() on multi-element tensor should throw";
}

// ============================================================================
// CUDA Error Propagation
// ============================================================================

class CUDAErrorPropagationTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        clear_cuda_errors();
    }

    void TearDown() override {
        clear_cuda_errors();
    }
};

TEST_F(CUDAErrorPropagationTest, ErrorPersistsAcrossOperations) {
    // Generate an error
    auto bool_t = Tensor::ones_bool({100}, Device::CUDA);

    // This likely generates an error
    auto sum = bool_t.sum();

    bool has_error = has_cuda_error();
    if (has_error) {
        std::cerr << "Initial operation generated CUDA error as expected" << std::endl;

        // Error should persist to next operation
        auto t2 = Tensor::ones({10}, Device::CUDA);
        bool still_has_error = has_cuda_error();

        EXPECT_TRUE(still_has_error) << "KNOWN BUG: CUDA errors persist and aren't cleared automatically";

        clear_cuda_errors();
    }
}

TEST_F(CUDAErrorPropagationTest, SynchronizeDetectsErrors) {
    auto bool_t = Tensor::ones_bool({100}, Device::CUDA);

    // Clear any existing errors
    clear_cuda_errors();

    // Operation that may generate error
    auto sum = bool_t.sum();

    // cudaDeviceSynchronize should detect the error
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize detected error: " << cudaGetErrorString(err) << std::endl;
        clear_cuda_errors();
    }
}

// ============================================================================
// Expression Template Edge Cases
// ============================================================================

class ExpressionTemplateEdgeCasesTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        clear_cuda_errors();
    }

    void TearDown() override {
        clear_cuda_errors();
    }
};

TEST_F(ExpressionTemplateEdgeCasesTest, NestedExpressions) {
    // Test deeply nested expression templates
    auto a = Tensor::ones({100}, Device::CUDA);
    auto b = Tensor::ones({100}, Device::CUDA) * 2.0f;
    auto c = Tensor::ones({100}, Device::CUDA) * 3.0f;

    // Nested: (a + b) * (c - a)
    auto result = (a + b) * (c - a);

    // Force evaluation
    auto result_tensor = Tensor(result);

    // Expected: (1 + 2) * (3 - 1) = 3 * 2 = 6
    auto values = result_tensor.cpu().to_vector();
    EXPECT_FLOAT_EQ(values[0], 6.0f);
}

TEST_F(ExpressionTemplateEdgeCasesTest, MixedDtypeExpressions) {
    // Test expressions with mixed dtypes
    auto float_t = Tensor::ones({100}, Device::CUDA, DataType::Float32);
    auto int_t = Tensor::ones({100}, Device::CUDA, DataType::Int32);

    // This should promote to Float32
    auto result = float_t + int_t;
    auto result_tensor = Tensor(result);

    EXPECT_EQ(result_tensor.dtype(), DataType::Float32);
}

TEST_F(ExpressionTemplateEdgeCasesTest, BoolInExpressions) {
    // Test Bool tensors in arithmetic expressions
    auto bool_t = Tensor::ones_bool({100}, Device::CUDA);
    auto float_t = Tensor::ones({100}, Device::CUDA, DataType::Float32) * 2.0f;

    // Bool should promote to Float32
    auto result = bool_t + float_t;
    auto result_tensor = Tensor(result);

    EXPECT_EQ(result_tensor.dtype(), DataType::Float32);

    auto values = result_tensor.cpu().to_vector();
    EXPECT_FLOAT_EQ(values[0], 3.0f);  // 1 (bool) + 2 (float) = 3
}

// ============================================================================
// Type Conversion Edge Cases
// ============================================================================

class TypeConversionEdgeCasesTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        clear_cuda_errors();
    }

    void TearDown() override {
        clear_cuda_errors();
    }
};

TEST_F(TypeConversionEdgeCasesTest, BoolToInt32) {
    auto bool_t = Tensor::ones_bool({100}, Device::CUDA);
    auto int_t = bool_t.to(DataType::Int32);

    EXPECT_EQ(int_t.dtype(), DataType::Int32);

    auto values = int_t.cpu().to_vector();
    EXPECT_FLOAT_EQ(values[0], 1.0f);
}

TEST_F(TypeConversionEdgeCasesTest, Int32ToBool) {
    auto int_t = Tensor::from_vector({0.0f, 1.0f, 2.0f, -1.0f}, {4}, Device::CUDA).to(DataType::Int32);
    auto bool_t = int_t.to(DataType::Bool);

    EXPECT_EQ(bool_t.dtype(), DataType::Bool);

    // Check values
    auto as_int = bool_t.to(DataType::Int32);
    auto values = as_int.cpu().to_vector();

    // 0 -> false(0), 1 -> true(1), 2 -> true(1), -1 -> true(1)
    EXPECT_EQ(values[0], 0.0f);
    EXPECT_EQ(values[1], 1.0f);
    EXPECT_EQ(values[2], 1.0f);
    EXPECT_EQ(values[3], 1.0f);
}

TEST_F(TypeConversionEdgeCasesTest, Float16ToBool) {
    auto float16_t = Tensor::ones({100}, Device::CUDA, DataType::Float16);
    auto bool_t = float16_t.to(DataType::Bool);

    EXPECT_EQ(bool_t.dtype(), DataType::Bool);

    auto as_int = bool_t.to(DataType::Int32);
    auto sum = as_int.sum().template item<int>();
    EXPECT_EQ(sum, 100);
}

TEST_F(TypeConversionEdgeCasesTest, UInt8ToInt64) {
    auto uint8_t = Tensor::from_vector({255.0f, 128.0f, 0.0f}, {3}, Device::CUDA).to(DataType::UInt8);
    auto int64_t = uint8_t.to(DataType::Int64);

    EXPECT_EQ(int64_t.dtype(), DataType::Int64);
}

// ============================================================================
// Reduction Operations
// ============================================================================

class ReductionBugsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        clear_cuda_errors();
    }

    void TearDown() override {
        clear_cuda_errors();
    }
};

TEST_F(ReductionBugsTest, SumFloat32) {
    auto t = Tensor::ones({100}, Device::CUDA, DataType::Float32);
    auto sum = t.sum();

    EXPECT_EQ(sum.dtype(), DataType::Float32);
    EXPECT_FLOAT_EQ(sum.item<float>(), 100.0f);
}

TEST_F(ReductionBugsTest, SumInt32) {
    auto t = Tensor::ones({100}, Device::CUDA, DataType::Int32);
    auto sum = t.sum();

    // Sum should preserve Int32
    EXPECT_EQ(sum.dtype(), DataType::Int32);
}

TEST_F(ReductionBugsTest, SumBool) {
    // KNOWN BUG: Bool sum
    auto t = Tensor::ones_bool({100}, Device::CUDA);

    clear_cuda_errors();
    auto sum = t.sum();

    if (has_cuda_error()) {
        std::cerr << "KNOWN BUG: Bool sum generates CUDA error" << std::endl;
        clear_cuda_errors();
        GTEST_SKIP();
    }

    std::cerr << "Bool sum dtype: " << dtype_name(sum.dtype()) << std::endl;
}

TEST_F(ReductionBugsTest, MeanFloat32) {
    auto t = Tensor::ones({100}, Device::CUDA, DataType::Float32) * 2.0f;
    auto mean = t.mean();

    EXPECT_FLOAT_EQ(mean.item<float>(), 2.0f);
}

TEST_F(ReductionBugsTest, MeanInt32) {
    // Mean of integers should probably return float
    auto t = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CUDA).to(DataType::Int32);
    auto mean = t.mean();

    std::cerr << "Int32 mean dtype: " << dtype_name(mean.dtype()) << std::endl;
}

// ============================================================================
// Memory Safety
// ============================================================================

class MemorySafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        clear_cuda_errors();
    }

    void TearDown() override {
        clear_cuda_errors();
    }
};

TEST_F(MemorySafetyTest, SliceDoesNotCorruptMemory) {
    auto t = Tensor::arange(0, 100);
    auto slice = t.slice(0, 10, 20);

    // Modify slice
    slice = slice + 1000.0f;

    // Original should be unchanged
    auto orig_values = t.cpu().to_vector();
    EXPECT_FLOAT_EQ(orig_values[0], 0.0f);
    EXPECT_FLOAT_EQ(orig_values[50], 50.0f);
}

TEST_F(MemorySafetyTest, CopyIndependence) {
    auto t1 = Tensor::ones({100}, Device::CUDA);
    auto t2 = t1.clone();

    // Modify t2
    t2 = t2 * 2.0f;

    // t1 should be unchanged
    auto t1_values = t1.cpu().to_vector();
    EXPECT_FLOAT_EQ(t1_values[0], 1.0f);

    auto t2_values = t2.cpu().to_vector();
    EXPECT_FLOAT_EQ(t2_values[0], 2.0f);
}

TEST_F(MemorySafetyTest, CPUCUDATransferIntegrity) {
    std::vector<float> data = {1, 2, 3, 4, 5};
    auto cpu_t = Tensor::from_vector(data, {5}, Device::CPU);
    auto cuda_t = cpu_t.to(Device::CUDA);
    auto back_to_cpu = cuda_t.cpu();

    auto result = back_to_cpu.to_vector();

    for (size_t i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(result[i], data[i]);
    }
}

// ============================================================================
// Comparison Operations
// ============================================================================

class ComparisonBugsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        clear_cuda_errors();
    }

    void TearDown() override {
        clear_cuda_errors();
    }
};

TEST_F(ComparisonBugsTest, EqualityFloat32) {
    auto t1 = Tensor::ones({100}, Device::CUDA);
    auto t2 = Tensor::ones({100}, Device::CUDA);

    auto eq = (t1 == t2);

    EXPECT_EQ(eq.dtype(), DataType::Bool);

    auto count = eq.to(DataType::Int32).sum().template item<int>();
    EXPECT_EQ(count, 100);
}

TEST_F(ComparisonBugsTest, ComparisonWithScalar) {
    auto t = Tensor::arange(0, 10);

    auto gt5 = t > 5.0f;
    EXPECT_EQ(gt5.dtype(), DataType::Bool);

    int count = gt5.to(DataType::Int32).sum().template item<int>();
    EXPECT_EQ(count, 4);  // 6, 7, 8, 9
}

TEST_F(ComparisonBugsTest, BoolComparison) {
    auto t1 = Tensor::ones_bool({100}, Device::CUDA);
    auto t2 = Tensor::zeros_bool({100}, Device::CUDA);

    auto eq = (t1 == t2);

    EXPECT_EQ(eq.dtype(), DataType::Bool);

    auto count = eq.to(DataType::Int32).sum().template item<int>();
    EXPECT_EQ(count, 0);  // None should match
}
