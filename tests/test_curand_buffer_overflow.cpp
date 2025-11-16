/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core_new/tensor.hpp"
#include <cuda_runtime.h>

using namespace lfs::core;

/**
 * @brief Test suite for curandGenerateNormal buffer overflow bug
 *
 * The bug: When tensor size is odd, Tensor::normal_() calls curandGenerateNormal
 * with n+1 elements but the buffer only has space for n elements. This writes
 * 1 element past the buffer, corrupting CUDA memory and causing crashes.
 *
 * Impact: This affects MCMC strategy inject_noise() which creates noise tensors
 * with odd sizes like [54275, 3] = 162,825 elements (ODD).
 */

class CurandBufferOverflowTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }

    // Helper to check if a size would trigger the bug
    static bool is_odd_size(size_t n) {
        return (n % 2) == 1;
    }
};

/**
 * Test 1: Small odd-sized tensor should trigger buffer overflow
 * This is the minimal reproduction case
 */
TEST_F(CurandBufferOverflowTest, SmallOddSizeTensor) {
    const size_t N = 5; // ODD size - will trigger bug
    ASSERT_TRUE(is_odd_size(N)) << "Test requires odd size";

    // Create tensor on CUDA
    Tensor t = Tensor::zeros({N}, Device::CUDA);

    // This should write 6 elements but buffer only has 5!
    // Without compute-sanitizer this may not crash, but corrupts memory
    EXPECT_NO_THROW({
        t.normal_(0.0f, 1.0f);
    }) << "normal_() should not throw (but may corrupt memory silently)";

    // Verify tensor is still valid after corruption
    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), N);
}

/**
 * Test 2: Realistic MCMC scenario - means tensor [N, 3]
 * This reproduces the exact bug from MCMC::inject_noise()
 */
TEST_F(CurandBufferOverflowTest, MCMCMeansTensorSize) {
    // Typical COLMAP point cloud has odd number of points
    const size_t N = 54275; // From bicycle dataset
    const size_t dims = 3;
    const size_t total = N * dims; // 162,825 - ODD!

    ASSERT_TRUE(is_odd_size(total)) << "MCMC means tensor must have odd total size";

    // Create means tensor [N, 3]
    Tensor means = Tensor::zeros({N, dims}, Device::CUDA);

    // Generate noise like MCMC does
    Tensor noise;
    EXPECT_NO_THROW({
        noise = Tensor::randn_like(means); // This calls normal_() internally
    });

    EXPECT_TRUE(noise.is_valid());
    EXPECT_EQ(noise.shape(), means.shape());
    EXPECT_EQ(noise.numel(), total);
}

/**
 * Test 3: Even-sized tensor should work correctly (control case)
 */
TEST_F(CurandBufferOverflowTest, EvenSizeTensorNoOverflow) {
    const size_t N = 6; // EVEN size - no bug
    ASSERT_FALSE(is_odd_size(N)) << "Control test requires even size";

    Tensor t = Tensor::zeros({N}, Device::CUDA);

    // This should work fine - generates exactly 6 elements
    EXPECT_NO_THROW({
        t.normal_(0.0f, 1.0f);
    });

    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), N);
}

/**
 * Test 4: Verify the bug with multiple consecutive allocations
 * Buffer overflow should corrupt subsequent allocations
 */
TEST_F(CurandBufferOverflowTest, CorruptsSubsequentAllocations) {
    const size_t N = 101; // ODD
    ASSERT_TRUE(is_odd_size(N));

    // Allocate first tensor
    Tensor t1 = Tensor::zeros({N}, Device::CUDA);

    // Allocate second tensor immediately after
    // If t1.normal_() overflows, it may corrupt t2's memory
    Tensor t2 = Tensor::zeros({10}, Device::CUDA);

    // Fill t1 with noise - THIS OVERFLOWS
    EXPECT_NO_THROW({
        t1.normal_(0.0f, 1.0f);
    });

    // Try to use t2 - may fail if its memory was corrupted
    EXPECT_NO_THROW({
        t2.fill_(42.0f);
    }) << "Subsequent tensor should still be usable";
}

/**
 * Test 5: Large odd-sized tensor (stress test)
 * Tests the bug with 4M Gaussians scenario
 */
TEST_F(CurandBufferOverflowTest, LargeOddSizeTensor) {
    const size_t N = 4000001; // 4M + 1 - ODD, large
    ASSERT_TRUE(is_odd_size(N));

    Tensor t = Tensor::zeros({N}, Device::CUDA);

    // This writes 4,000,002 elements but buffer has 4,000,001
    EXPECT_NO_THROW({
        t.normal_(0.0f, 1.0f);
    });

    EXPECT_TRUE(t.is_valid());
}

/**
 * Test 6: Verify buffer overflow with compute-sanitizer detection
 * This test documents what compute-sanitizer should report
 */
TEST_F(CurandBufferOverflowTest, DocumentedBugBehavior) {
    // From compute-sanitizer output:
    // "Invalid __global__ write of size 4 bytes"
    // "Address is out of bounds and is 1 bytes after nearest allocation"
    // "at curandGenerateNormal"

    const size_t N = 163; // ODD - matches compute-sanitizer report pattern
    ASSERT_TRUE(is_odd_size(N));

    Tensor t = Tensor::zeros({N}, Device::CUDA);

    // When run with compute-sanitizer, this should report:
    // - Out of bounds write
    // - 1 element past allocated buffer
    // - Caused by curandGenerateNormal
    t.normal_(0.0f, 1.0f);

    // Note: Without compute-sanitizer, test may pass despite memory corruption
    EXPECT_TRUE(t.is_valid()) << "Tensor may appear valid despite corruption";
}

/**
 * Test 7: randn_like should preserve shape exactly
 */
TEST_F(CurandBufferOverflowTest, RandnLikePreservesShape) {
    // Test various odd shapes
    std::vector<std::vector<size_t>> odd_shapes = {
        {5},           // 5 - odd
        {3, 3},        // 9 - odd
        {11, 7},       // 77 - odd
        {101, 3},      // 303 - odd
        {54275, 3},    // 162,825 - odd (MCMC real case)
    };

    for (const auto& shape_vec : odd_shapes) {
        size_t total = 1;
        for (auto dim : shape_vec) total *= dim;

        if (!is_odd_size(total)) continue; // Skip even cases

        TensorShape shape(shape_vec);
        Tensor original = Tensor::zeros(shape, Device::CUDA);
        Tensor noise = Tensor::randn_like(original);

        EXPECT_EQ(noise.shape(), original.shape())
            << "randn_like must preserve shape";
        EXPECT_EQ(noise.numel(), total)
            << "Total elements must match for size " << total;
    }
}

/**
 * Test 8: Verify the fix (when implemented)
 * After fix, this test should validate correct behavior
 */
TEST_F(CurandBufferOverflowTest, VerifyFixAllocatesExtraSpace) {
    const size_t N = 999; // ODD
    ASSERT_TRUE(is_odd_size(N));

    Tensor t = Tensor::zeros({N}, Device::CUDA);

    // After fix, internal allocation should reserve space for N+1 elements
    // when N is odd, to accommodate curandGenerateNormal's requirement
    EXPECT_NO_THROW({
        t.normal_(0.0f, 1.0f);
    });

    // Values should be filled correctly
    EXPECT_TRUE(t.is_valid());

    // Only first N values should be used (last element ignored)
    EXPECT_EQ(t.numel(), N) << "Public API should still report N elements";
}

/**
 * Test 9: Check impact on CUDA error state
 * Buffer overflow causes all subsequent CUDA operations to fail
 */
TEST_F(CurandBufferOverflowTest, CorruptsCUDAErrorState) {
    const size_t N = 51; // ODD
    ASSERT_TRUE(is_odd_size(N));

    // Clear any previous CUDA errors
    cudaGetLastError();

    Tensor t = Tensor::zeros({N}, Device::CUDA);
    t.normal_(0.0f, 1.0f); // May cause buffer overflow

    // Try a simple CUDA operation
    Tensor t2 = Tensor::zeros({10}, Device::CUDA);

    // Check CUDA error state
    // If buffer overflow occurred, CUDA may be in error state
    cudaError_t err = cudaGetLastError();

    // Note: Without compute-sanitizer, this may still be cudaSuccess
    // because the corruption is silent
    if (err != cudaSuccess) {
        ADD_FAILURE() << "CUDA entered error state: " << cudaGetErrorString(err);
    }
}

/**
 * Test 10: Memory leak detection
 * Corrupted memory may not be properly freed
 */
TEST_F(CurandBufferOverflowTest, NoMemoryLeakAfterCorruption) {
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);

    {
        const size_t N = 10001; // ODD
        Tensor t = Tensor::zeros({N}, Device::CUDA);
        t.normal_(0.0f, 1.0f);

        // Tensor goes out of scope - should free memory
    }

    // Force cleanup
    cudaDeviceSynchronize();

    size_t free_after;
    cudaMemGetInfo(&free_after, &total);

    // Memory should be recovered (within tolerance)
    const size_t tolerance = 1 << 20; // 1MB
    EXPECT_NEAR(free_after, free_before, tolerance)
        << "Memory should be freed after tensor destruction";
}
