/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core_new/tensor.hpp"
#include <vector>

using namespace lfs::core;

// ===================================================================================
// Bool Tensor Conversion Tests
// ===================================================================================

TEST(TensorBoolTest, Int32ToBoolConversion) {
    // Create int32 tensor and convert to bool
    std::vector<int32_t> int_vec = {0, 1, 0, 1, 1};
    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({5}), Device::CUDA);
    auto bool_tensor = int_tensor.to(DataType::Bool);
    
    // Convert back to int32 to verify
    auto back_to_int = bool_tensor.to(DataType::Int32);
    auto result_vec = back_to_int.cpu().to_vector();
    
    EXPECT_EQ(result_vec.size(), 5);
    EXPECT_FLOAT_EQ(result_vec[0], 0.0f);
    EXPECT_FLOAT_EQ(result_vec[1], 1.0f);
    EXPECT_FLOAT_EQ(result_vec[2], 0.0f);
    EXPECT_FLOAT_EQ(result_vec[3], 1.0f);
    EXPECT_FLOAT_EQ(result_vec[4], 1.0f);
}

TEST(TensorBoolTest, BoolTensorSum) {
    // Create bool tensor and sum it
    std::vector<int32_t> int_vec = {0, 1, 0, 1, 1};
    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({5}), Device::CUDA);
    auto bool_tensor = int_tensor.to(DataType::Bool);

    // Sum should count the number of true values
    auto bool_as_int = bool_tensor.to(DataType::Int32);
    int sum = bool_as_int.sum().item<int>();

    EXPECT_EQ(sum, 3) << "Bool tensor should have 3 true values";
}

TEST(TensorBoolTest, BoolZerosThenFill) {
    // Create bool zeros then try to fill
    auto bool_zeros = Tensor::zeros({10}, Device::CUDA, DataType::Bool);
    
    // Try to fill a slice with true
    bool_zeros.slice(0, 2, 7).fill_(true);
    
    // Check result
    auto result = bool_zeros.to(DataType::Int32).cpu().to_vector();
    
    EXPECT_EQ(result.size(), 10);
    EXPECT_FLOAT_EQ(result[0], 0.0f);  // Before slice
    EXPECT_FLOAT_EQ(result[1], 0.0f);  // Before slice
    EXPECT_FLOAT_EQ(result[2], 1.0f);  // In slice
    EXPECT_FLOAT_EQ(result[3], 1.0f);  // In slice
    EXPECT_FLOAT_EQ(result[4], 1.0f);  // In slice
    EXPECT_FLOAT_EQ(result[5], 1.0f);  // In slice
    EXPECT_FLOAT_EQ(result[6], 1.0f);  // In slice
    EXPECT_FLOAT_EQ(result[7], 0.0f);  // After slice
    EXPECT_FLOAT_EQ(result[8], 0.0f);  // After slice
    EXPECT_FLOAT_EQ(result[9], 0.0f);  // After slice
}

TEST(TensorBoolTest, BoolVectorDirectCreation) {
    // Try to create bool tensor directly from bool vector
    std::vector<bool> bool_vec = {false, true, false, true, true};

    auto bool_tensor = Tensor::from_vector(bool_vec, TensorShape({5}), Device::CUDA);
    auto result = bool_tensor.to(DataType::Int32).cpu().to_vector();

    EXPECT_EQ(result.size(), 5);
    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 1.0f);
    EXPECT_FLOAT_EQ(result[2], 0.0f);
    EXPECT_FLOAT_EQ(result[3], 1.0f);
    EXPECT_FLOAT_EQ(result[4], 1.0f);
}

TEST(TensorBoolTest, LogicalNot) {
    // Test logical_not operation
    std::vector<int32_t> int_vec = {0, 1, 0, 1, 1};
    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({5}), Device::CUDA);
    auto bool_tensor = int_tensor.to(DataType::Bool);
    
    auto bool_not = bool_tensor.logical_not();
    auto result = bool_not.to(DataType::Int32).cpu().to_vector();
    
    EXPECT_EQ(result.size(), 5);
    EXPECT_FLOAT_EQ(result[0], 1.0f);  // NOT 0 = 1
    EXPECT_FLOAT_EQ(result[1], 0.0f);  // NOT 1 = 0
    EXPECT_FLOAT_EQ(result[2], 1.0f);  // NOT 0 = 1
    EXPECT_FLOAT_EQ(result[3], 0.0f);  // NOT 1 = 0
    EXPECT_FLOAT_EQ(result[4], 0.0f);  // NOT 1 = 0
}

TEST(TensorBoolTest, LogicalOr) {
    // Test logical_or operation
    std::vector<int32_t> vec1 = {0, 1, 0, 1};
    std::vector<int32_t> vec2 = {0, 0, 1, 1};
    
    auto tensor1 = Tensor::from_vector(vec1, TensorShape({4}), Device::CUDA).to(DataType::Bool);
    auto tensor2 = Tensor::from_vector(vec2, TensorShape({4}), Device::CUDA).to(DataType::Bool);
    
    auto result_tensor = tensor1.logical_or(tensor2);
    auto result = result_tensor.to(DataType::Int32).cpu().to_vector();
    
    EXPECT_EQ(result.size(), 4);
    EXPECT_FLOAT_EQ(result[0], 0.0f);  // 0 OR 0 = 0
    EXPECT_FLOAT_EQ(result[1], 1.0f);  // 1 OR 0 = 1
    EXPECT_FLOAT_EQ(result[2], 1.0f);  // 0 OR 1 = 1
    EXPECT_FLOAT_EQ(result[3], 1.0f);  // 1 OR 1 = 1
}

TEST(TensorBoolTest, LogicalAnd) {
    // Test logical_and operation
    std::vector<int32_t> vec1 = {0, 1, 0, 1};
    std::vector<int32_t> vec2 = {0, 0, 1, 1};
    
    auto tensor1 = Tensor::from_vector(vec1, TensorShape({4}), Device::CUDA).to(DataType::Bool);
    auto tensor2 = Tensor::from_vector(vec2, TensorShape({4}), Device::CUDA).to(DataType::Bool);
    
    auto result_tensor = tensor1.logical_and(tensor2);
    auto result = result_tensor.to(DataType::Int32).cpu().to_vector();
    
    EXPECT_EQ(result.size(), 4);
    EXPECT_FLOAT_EQ(result[0], 0.0f);  // 0 AND 0 = 0
    EXPECT_FLOAT_EQ(result[1], 0.0f);  // 1 AND 0 = 0
    EXPECT_FLOAT_EQ(result[2], 0.0f);  // 0 AND 1 = 0
    EXPECT_FLOAT_EQ(result[3], 1.0f);  // 1 AND 1 = 1
}

TEST(TensorBoolTest, NonzeroOnBoolTensor) {
    // Test nonzero() on bool tensor
    std::vector<int32_t> int_vec = {0, 1, 0, 1, 1, 0, 1};
    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({7}), Device::CUDA);
    auto bool_tensor = int_tensor.to(DataType::Bool);
    
    auto indices = bool_tensor.nonzero();
    
    // Should return a 2D tensor of shape [num_nonzero, 1] for 1D input
    EXPECT_EQ(indices.ndim(), 2);
    
    // Squeeze to get 1D indices
    auto indices_1d = indices.squeeze(-1);
    auto result = indices_1d.cpu().to_vector();
    
    // Should find indices: 1, 3, 4, 6
    EXPECT_EQ(result.size(), 4);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 3.0f);
    EXPECT_FLOAT_EQ(result[2], 4.0f);
    EXPECT_FLOAT_EQ(result[3], 6.0f);
}

TEST(TensorBoolTest, BoolIndexing) {
    // Test using bool tensor for indexing
    auto data = Tensor::from_vector(std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f},
                                     TensorShape({5}), Device::CUDA);

    std::vector<int32_t> mask_vec = {1, 0, 1, 0, 1};
    auto mask_int = Tensor::from_vector(mask_vec, TensorShape({5}), Device::CUDA);
    auto mask = mask_int.to(DataType::Bool);

    // Use index_select with nonzero indices
    auto indices = mask.nonzero().squeeze(-1);
    auto selected = data.index_select(0, indices);

    auto result = selected.cpu().to_vector();
    EXPECT_EQ(result.size(), 3);
    EXPECT_FLOAT_EQ(result[0], 10.0f);
    EXPECT_FLOAT_EQ(result[1], 30.0f);
    EXPECT_FLOAT_EQ(result[2], 50.0f);
}

TEST(TensorBoolTest, BoolComparisonResult) {
    // Test that comparison operations return proper bool tensors
    auto tensor = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                                      TensorShape({5}), Device::CUDA);
    
    auto mask = tensor > 3.0f;  // Should be bool tensor
    
    // Convert to int to check
    auto result = mask.to(DataType::Int32).cpu().to_vector();
    
    EXPECT_EQ(result.size(), 5);
    EXPECT_FLOAT_EQ(result[0], 0.0f);  // 1 > 3 = false
    EXPECT_FLOAT_EQ(result[1], 0.0f);  // 2 > 3 = false
    EXPECT_FLOAT_EQ(result[2], 0.0f);  // 3 > 3 = false
    EXPECT_FLOAT_EQ(result[3], 1.0f);  // 4 > 3 = true
    EXPECT_FLOAT_EQ(result[4], 1.0f);  // 5 > 3 = true
}

TEST(TensorBoolTest, MCMCRemoveGaussiansScenario) {
    // Reproduce the exact scenario from MCMC remove_gaussians
    const int N = 100;

    // Create mask to remove indices 10-39 (30 elements)
    std::vector<int32_t> mask_vec(N, 0);
    for (int i = 10; i < 40; i++) {
        mask_vec[i] = 1;
    }

    auto mask_int = Tensor::from_vector(mask_vec, TensorShape({N}), Device::CUDA);
    auto mask = mask_int.to(DataType::Bool);

    // Convert to int and sum (this is what MCMC does)
    auto mask_int_back = mask.to(DataType::Int32);
    int n_remove = mask_int_back.sum().item<int>();

    EXPECT_EQ(n_remove, 30) << "Should detect 30 Gaussians to remove";

    // Get keep indices
    auto keep_mask = mask.logical_not();
    auto keep_indices = keep_mask.nonzero().squeeze(-1);

    EXPECT_EQ(keep_indices.numel(), 70) << "Should have 70 Gaussians to keep";
}

// ===================================================================================
// Bool Reduction Kernel Tests (NEW - tests launch_reduce_op_bool directly)
// ===================================================================================

TEST(BoolReductionKernel, SumScalarAllTrue) {
    // Create tensor with all True values
    auto t = Tensor::full({10000}, true, Device::CUDA, DataType::Bool);
    float sum = t.sum_scalar();
    EXPECT_FLOAT_EQ(sum, 10000.0f) << "Should count all True values";
}

TEST(BoolReductionKernel, SumScalarAllFalse) {
    // Create tensor with all False values
    auto t = Tensor::full({10000}, false, Device::CUDA, DataType::Bool);
    float sum = t.sum_scalar();
    EXPECT_FLOAT_EQ(sum, 0.0f) << "Should count zero True values";
}

TEST(BoolReductionKernel, SumScalarMixed) {
    // Create tensor with mixed true/false
    std::vector<int32_t> int_vec(10000, 1);
    for (int i = 0; i < 5000; i++) {
        int_vec[i] = 0;  // First half false
    }

    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({10000}), Device::CUDA);
    auto bool_tensor = int_tensor.to(DataType::Bool);

    float sum = bool_tensor.sum_scalar();
    EXPECT_FLOAT_EQ(sum, 5000.0f) << "Should count 5000 True values";
}

TEST(BoolReductionKernel, LargeSum10M) {
    // Test with 10M elements (same scale as the densification bug)
    auto t = Tensor::full({10000000}, true, Device::CUDA, DataType::Bool);
    float sum = t.sum_scalar();
    EXPECT_FLOAT_EQ(sum, 10000000.0f) << "Should handle 10M elements without corruption";
}

TEST(BoolReductionKernel, MeanOperation) {
    // Mean on Bool: convert to Int32 first for meaningful result
    std::vector<int32_t> int_vec(1000, 1);
    for (int i = 0; i < 500; i++) {
        int_vec[i] = 0;  // First half false
    }

    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({1000}), Device::CUDA);
    auto bool_tensor = int_tensor.to(DataType::Bool);

    // Convert to Int32 for mean operation (returns int with integer division)
    auto as_int = bool_tensor.to(DataType::Int32);
    auto result = as_int.mean();
    int mean = result.item<int>();
    EXPECT_EQ(mean, 0) << "Mean with int division should be 0 (500/1000=0)";
}

TEST(BoolReductionKernel, MaxOperation) {
    // Max on Bool: convert to Int32 first
    auto t_all_false = Tensor::full({1000}, false, Device::CUDA, DataType::Bool);
    auto max_false = t_all_false.to(DataType::Int32).max();
    EXPECT_EQ(max_false.item<int>(), 0) << "Max of all false should be 0";

    std::vector<int32_t> int_vec(1000, 0);
    int_vec[500] = 1;  // Set one to true
    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({1000}), Device::CUDA);
    auto t_some_true = int_tensor.to(DataType::Bool);

    auto max_true = t_some_true.to(DataType::Int32).max();
    EXPECT_EQ(max_true.item<int>(), 1) << "Max with any true should be 1";
}

TEST(BoolReductionKernel, MinOperation) {
    // Min on Bool: convert to Int32 first
    auto t_all_true = Tensor::full({1000}, true, Device::CUDA, DataType::Bool);
    auto min_true = t_all_true.to(DataType::Int32).min();
    EXPECT_EQ(min_true.item<int>(), 1) << "Min of all true should be 1";

    std::vector<int32_t> int_vec(1000, 1);
    int_vec[500] = 0;  // Set one to false
    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({1000}), Device::CUDA);
    auto t_some_false = int_tensor.to(DataType::Bool);

    auto min_false = t_some_false.to(DataType::Int32).min();
    EXPECT_EQ(min_false.item<int>(), 0) << "Min with any false should be 0";
}

TEST(BoolReductionKernel, ComparisonResultSum) {
    // Test that comparison results can be summed correctly
    auto t = Tensor::arange(0, 100, 1);  // Creates Float32 tensor on CUDA by default
    auto mask = t > 50.0f;  // Should create Bool tensor

    EXPECT_EQ(mask.dtype(), DataType::Bool);

    float count = mask.sum_scalar();
    EXPECT_FLOAT_EQ(count, 49.0f) << "Should count values 51-99 (49 values)";
}

TEST(BoolReductionKernel, ZerosTensorBugFix) {
    // This is the specific bug that was happening in densification:
    // zeros.sum_scalar() was returning 1065353216 (garbage)
    auto zeros = Tensor::zeros({1000000}, Device::CUDA, DataType::Bool);
    float sum = zeros.sum_scalar();

    // Before fix: returned 1065353216 (float bits interpreted as int)
    // After fix: should return 0
    EXPECT_FLOAT_EQ(sum, 0.0f) << "Bool zeros sum should be 0, not garbage";
}

TEST(BoolReductionKernel, DensificationNumDuplicatesBugFix) {
    // Exact scenario from densification where num_duplicates was corrupted
    const int N = 5000000;  // 5M elements

    // Create all-zeros bool tensor (simulating no duplicates)
    auto zeros = Tensor::zeros({N}, Device::CUDA, DataType::Bool);

    // This was returning 1065353216 before the fix
    float num_duplicates = zeros.sum_scalar();

    EXPECT_FLOAT_EQ(num_duplicates, 0.0f) << "num_duplicates should be 0 when no duplicates exist";
    EXPECT_LT(num_duplicates, 100.0f) << "num_duplicates should be reasonable, not 1065353216";
}

TEST(BoolReductionKernel, DirectSumWithoutConversion) {
    // Verify we can sum Bool tensors directly without converting to Int32 first
    std::vector<int32_t> int_vec = {0, 1, 0, 1, 1, 0, 1, 1};
    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({8}), Device::CUDA);
    auto bool_tensor = int_tensor.to(DataType::Bool);

    // Direct sum on Bool tensor (returns Int64)
    auto sum_result = bool_tensor.sum();
    int64_t sum = sum_result.item<int64_t>();

    EXPECT_EQ(sum, 5) << "Should directly sum Bool tensor to count True values";
}

