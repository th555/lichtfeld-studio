/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor/tensor.hpp"
#include <gtest/gtest.h>

using namespace lfs::core;

TEST(BoolReduction, SumScalarAllTrue) {
    // Create tensor with all True values
    auto t = Tensor::full({10000}, true, DataType::Bool, Device::CUDA);
    float sum = t.sum_scalar();
    ASSERT_FLOAT_EQ(sum, 10000.0f);  // Should count all trues
}

TEST(BoolReduction, SumScalarAllFalse) {
    // Create tensor with all False values
    auto t = Tensor::full({10000}, false, DataType::Bool, Device::CUDA);
    float sum = t.sum_scalar();
    ASSERT_FLOAT_EQ(sum, 0.0f);  // Should count zero trues
}

TEST(BoolReduction, SumScalarMixed) {
    // Create tensor with mixed true/false
    auto t = Tensor::full({100, 100}, true, DataType::Bool, Device::CUDA);

    // Set half to false
    auto half = t.slice(0, 0, 50);  // First 50 rows
    half.fill_(false);

    float sum = t.sum_scalar();
    ASSERT_FLOAT_EQ(sum, 5000.0f);  // Should count 5000 trues (50*100)
}

TEST(BoolReduction, LargeSum) {
    // Test with 10M elements (same scale as the densification bug)
    auto t = Tensor::full({10000000}, true, DataType::Bool, Device::CUDA);
    float sum = t.sum_scalar();
    ASSERT_FLOAT_EQ(sum, 10000000.0f);
}

TEST(BoolReduction, MeanOperation) {
    // Test mean (should divide by count)
    auto t = Tensor::full({1000}, true, DataType::Bool, Device::CUDA);

    // Set half to false
    auto half = t.slice(0, 0, 500);
    half.fill_(false);

    auto result = t.mean();
    float mean = result.item<float>();
    ASSERT_FLOAT_EQ(mean, 0.5f);  // 500/1000 = 0.5
}

TEST(BoolReduction, MaxOperation) {
    // Max of bools: 1 if any true, 0 if all false
    auto t_all_false = Tensor::full({1000}, false, DataType::Bool, Device::CUDA);
    auto max_false = t_all_false.max();
    ASSERT_FLOAT_EQ(max_false.item<float>(), 0.0f);

    auto t_some_true = Tensor::full({1000}, false, DataType::Bool, Device::CUDA);
    t_some_true[500] = true;  // Set one element to true
    auto max_true = t_some_true.max();
    ASSERT_FLOAT_EQ(max_true.item<float>(), 1.0f);
}

TEST(BoolReduction, MinOperation) {
    // Min of bools: 0 if any false, 1 if all true
    auto t_all_true = Tensor::full({1000}, true, DataType::Bool, Device::CUDA);
    auto min_true = t_all_true.min();
    ASSERT_FLOAT_EQ(min_true.item<float>(), 1.0f);

    auto t_some_false = Tensor::full({1000}, true, DataType::Bool, Device::CUDA);
    t_some_false[500] = false;  // Set one element to false
    auto min_false = t_some_false.min();
    ASSERT_FLOAT_EQ(min_false.item<float>(), 0.0f);
}

TEST(BoolReduction, ComparisonResult) {
    // Test that comparison results can be summed correctly
    auto t = Tensor::arange(0, 100, 1, DataType::Float32, Device::CUDA);
    auto mask = t > 50.0f;  // Should create Bool tensor

    ASSERT_EQ(mask.dtype(), DataType::Bool);

    float count = mask.sum_scalar();
    ASSERT_FLOAT_EQ(count, 49.0f);  // Values 51-99 (49 values)
}

TEST(BoolReduction, NoDuplicatesBugFix) {
    // This is the specific bug case: zeros.sum_scalar() was returning garbage
    auto zeros = Tensor::zeros({1000000}, DataType::Bool, Device::CUDA);
    float sum = zeros.sum_scalar();

    // Before fix: returned 1065353216 (garbage, looks like float bits interpreted as int)
    // After fix: should return 0
    ASSERT_FLOAT_EQ(sum, 0.0f);
}
