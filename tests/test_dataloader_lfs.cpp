/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include "training_new/dataset.hpp"
#include "training/dataset.hpp"  // For comparison benchmarks only

using namespace std::chrono;
using namespace lfs::training;

// ===================================================================================
// Unit Tests - Thread-Safe Queue
// ===================================================================================

TEST(ThreadSafeQueueTest, PushPop) {
    ThreadSafeQueue<int> queue;

    queue.push(1);
    queue.push(2);
    queue.push(3);

    auto v1 = queue.pop();
    ASSERT_TRUE(v1.has_value());
    EXPECT_EQ(*v1, 1);

    auto v2 = queue.pop();
    ASSERT_TRUE(v2.has_value());
    EXPECT_EQ(*v2, 2);

    auto v3 = queue.pop();
    ASSERT_TRUE(v3.has_value());
    EXPECT_EQ(*v3, 3);
}

TEST(ThreadSafeQueueTest, Clear) {
    ThreadSafeQueue<int> queue;

    queue.push(1);
    queue.push(2);
    queue.push(3);

    size_t cleared = queue.clear();
    EXPECT_EQ(cleared, 3);
}

TEST(ThreadSafeQueueTest, Timeout) {
    ThreadSafeQueue<int> queue;

    // Try to pop with timeout - should return nullopt
    auto result = queue.pop(std::chrono::milliseconds(100));
    EXPECT_FALSE(result.has_value());
}

// ===================================================================================
// Unit Tests - Random Sampler
// ===================================================================================

TEST(RandomSamplerTest, BasicSampling) {
    RandomSampler sampler(10);

    std::vector<size_t> all_indices;

    // Get all indices in batches
    while (auto batch = sampler.next(3)) {
        all_indices.insert(all_indices.end(), batch->begin(), batch->end());
    }

    // Should have exactly 10 indices total
    EXPECT_EQ(all_indices.size(), 10);

    // All indices should be unique
    std::sort(all_indices.begin(), all_indices.end());
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(all_indices[i], i);
    }
}

TEST(RandomSamplerTest, Reset) {
    RandomSampler sampler(10);

    // Exhaust sampler
    while (sampler.next(3)) {}

    // Should return nullopt when exhausted
    EXPECT_FALSE(sampler.next(1).has_value());

    // Reset and try again
    sampler.reset();
    auto batch = sampler.next(1);
    EXPECT_TRUE(batch.has_value());
    EXPECT_EQ(batch->size(), 1);
}

TEST(RandomSamplerTest, ResizeOnReset) {
    RandomSampler sampler(10);

    sampler.reset(20);

    std::vector<size_t> all_indices;
    while (auto batch = sampler.next(5)) {
        all_indices.insert(all_indices.end(), batch->begin(), batch->end());
    }

    // Should have 20 indices after resize
    EXPECT_EQ(all_indices.size(), 20);
}

// ===================================================================================
// Unit Tests - Infinite Random Sampler
// ===================================================================================

TEST(InfiniteRandomSamplerTest, NeverExhausts) {
    InfiniteRandomSampler sampler(10);

    // Sample 30 items (3 full epochs)
    size_t total_sampled = 0;
    for (int i = 0; i < 30; ++i) {
        auto batch = sampler.next(1);
        ASSERT_TRUE(batch.has_value());
        total_sampled += batch->size();
    }

    EXPECT_EQ(total_sampled, 30);
}

TEST(InfiniteRandomSamplerTest, AutoReset) {
    InfiniteRandomSampler sampler(5);

    std::vector<size_t> first_epoch;
    std::vector<size_t> second_epoch;

    // Get first epoch
    for (int i = 0; i < 5; ++i) {
        auto batch = sampler.next(1);
        ASSERT_TRUE(batch.has_value());
        first_epoch.insert(first_epoch.end(), batch->begin(), batch->end());
    }

    // Get second epoch (should auto-reset)
    for (int i = 0; i < 5; ++i) {
        auto batch = sampler.next(1);
        ASSERT_TRUE(batch.has_value());
        second_epoch.insert(second_epoch.end(), batch->begin(), batch->end());
    }

    EXPECT_EQ(first_epoch.size(), 5);
    EXPECT_EQ(second_epoch.size(), 5);

    // Both epochs should be shuffled differently (with high probability)
    EXPECT_NE(first_epoch, second_epoch);
}

// ===================================================================================
// Unit Tests - DataLoader Options
// ===================================================================================

TEST(DataLoaderOptionsTest, Defaults) {
    DataLoaderOptions opts;

    EXPECT_EQ(opts.batch_size, 1);
    EXPECT_EQ(opts.num_workers, 0);
    EXPECT_EQ(opts.max_jobs, 0);
    EXPECT_TRUE(opts.enforce_ordering);
    EXPECT_FALSE(opts.drop_last);
}

// ===================================================================================
// Unit Tests - Dataset Structure
// ===================================================================================

TEST(CameraWithImageTest, Structure) {
    // Just verify the struct compiles
    CameraWithImage cwi;
    cwi.camera = nullptr;
    cwi.image = lfs::core::Tensor();

    EXPECT_EQ(cwi.camera, nullptr);
}

TEST(CameraExampleTest, Structure) {
    CameraExample example;
    example.data.camera = nullptr;
    example.data.image = lfs::core::Tensor();
    example.target = lfs::core::Tensor();

    EXPECT_EQ(example.data.camera, nullptr);
}

// ===================================================================================
// Performance Tests - Component Level
// ===================================================================================

TEST(DataLoaderPerf, RandomSampler_10k_Items) {
    const size_t dataset_size = 10000;

    auto start = high_resolution_clock::now();

    RandomSampler sampler(dataset_size);
    size_t total = 0;
    while (auto batch = sampler.next(1)) {
        total += batch->size();
    }

    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<microseconds>(end - start).count();

    EXPECT_EQ(total, dataset_size);
    std::cout << "  RandomSampler (10k items): " << elapsed << " us, "
              << (total * 1000000.0 / elapsed) << " items/sec" << std::endl;
}

TEST(DataLoaderPerf, InfiniteSampler_30k_Samples) {
    const size_t dataset_size = 100;
    const size_t num_samples = 30000;

    auto start = high_resolution_clock::now();

    InfiniteRandomSampler sampler(dataset_size);
    size_t total = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        auto batch = sampler.next(1);
        if (batch) total += batch->size();
    }

    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<microseconds>(end - start).count();

    EXPECT_EQ(total, num_samples);
    std::cout << "  InfiniteSampler (30k samples): " << elapsed << " us, "
              << (total * 1000000.0 / elapsed) << " items/sec" << std::endl;
}

TEST(DataLoaderPerf, ThreadSafeQueue_10k_Ops) {
    ThreadSafeQueue<int> queue;
    const size_t num_ops = 10000;

    auto start = high_resolution_clock::now();

    // Push
    for (size_t i = 0; i < num_ops; ++i) {
        queue.push(i);
    }

    // Pop
    size_t count = 0;
    for (size_t i = 0; i < num_ops; ++i) {
        if (queue.pop(std::chrono::milliseconds(1000))) {
            count++;
        }
    }

    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<microseconds>(end - start).count();

    EXPECT_EQ(count, num_ops);
    std::cout << "  ThreadSafeQueue (10k ops): " << elapsed << " us, "
              << (count * 2 * 1000000.0 / elapsed) << " ops/sec" << std::endl;
}

// ===================================================================================
// Comparison Benchmarks - LibTorch vs LibTorch-Free
// ===================================================================================

class DataLoaderComparison : public ::testing::Test {
protected:
    void print_comparison(const std::string& test_name,
                         double libtorch_us, double lfs_us, size_t count) {
        double speedup = libtorch_us / lfs_us;
        double libtorch_throughput = count * 1000000.0 / libtorch_us;
        double lfs_throughput = count * 1000000.0 / lfs_us;

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  " << std::left << std::setw(35) << test_name
                  << "LibTorch: " << std::setw(10) << libtorch_us << " us  "
                  << "LFS: " << std::setw(10) << lfs_us << " us  "
                  << "Speedup: " << std::setw(6) << speedup << "x" << std::endl;
    }
};

TEST_F(DataLoaderComparison, RandomSampler_SingleEpoch) {
    const size_t dataset_size = 10000;
    const size_t batch_size = 1;

    // LibTorch
    auto start1 = high_resolution_clock::now();
    torch::data::samplers::RandomSampler sampler1(dataset_size);
    size_t total1 = 0;
    while (auto batch = sampler1.next(batch_size)) {
        total1 += batch->size();
    }
    auto end1 = high_resolution_clock::now();
    auto elapsed1 = duration_cast<microseconds>(end1 - start1).count();

    // LibTorch-Free
    auto start2 = high_resolution_clock::now();
    lfs::training::RandomSampler sampler2(dataset_size);
    size_t total2 = 0;
    while (auto batch = sampler2.next(batch_size)) {
        total2 += batch->size();
    }
    auto end2 = high_resolution_clock::now();
    auto elapsed2 = duration_cast<microseconds>(end2 - start2).count();

    EXPECT_EQ(total1, dataset_size);
    EXPECT_EQ(total2, dataset_size);

    print_comparison("Single Epoch (10k items)", elapsed1, elapsed2, total1);
}

TEST_F(DataLoaderComparison, InfiniteSampler_10k_Samples) {
    const size_t dataset_size = 100;
    const size_t num_samples = 10000;
    const size_t batch_size = 1;

    // LibTorch
    auto start1 = high_resolution_clock::now();
    gs::training::InfiniteRandomSampler sampler1(dataset_size);
    size_t total1 = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        auto batch = sampler1.next(batch_size);
        if (batch) total1 += batch->size();
    }
    auto end1 = high_resolution_clock::now();
    auto elapsed1 = duration_cast<microseconds>(end1 - start1).count();

    // LibTorch-Free
    auto start2 = high_resolution_clock::now();
    lfs::training::InfiniteRandomSampler sampler2(dataset_size);
    size_t total2 = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        auto batch = sampler2.next(batch_size);
        if (batch) total2 += batch->size();
    }
    auto end2 = high_resolution_clock::now();
    auto elapsed2 = duration_cast<microseconds>(end2 - start2).count();

    EXPECT_EQ(total1, num_samples);
    EXPECT_EQ(total2, num_samples);

    print_comparison("Infinite Sampler (10k samples)", elapsed1, elapsed2, total1);
}

TEST_F(DataLoaderComparison, LargeBatches) {
    const size_t dataset_size = 10000;
    const size_t batch_size = 32;

    // LibTorch
    auto start1 = high_resolution_clock::now();
    torch::data::samplers::RandomSampler sampler1(dataset_size);
    size_t total1 = 0;
    while (auto batch = sampler1.next(batch_size)) {
        total1 += batch->size();
    }
    auto end1 = high_resolution_clock::now();
    auto elapsed1 = duration_cast<microseconds>(end1 - start1).count();

    // LibTorch-Free
    auto start2 = high_resolution_clock::now();
    lfs::training::RandomSampler sampler2(dataset_size);
    size_t total2 = 0;
    while (auto batch = sampler2.next(batch_size)) {
        total2 += batch->size();
    }
    auto end2 = high_resolution_clock::now();
    auto elapsed2 = duration_cast<microseconds>(end2 - start2).count();

    print_comparison("Large Batches (batch=32)", elapsed1, elapsed2, total1);
}

TEST_F(DataLoaderComparison, LargeDataset_1M) {
    const size_t dataset_size = 1000000;
    const size_t batch_size = 1;

    // LibTorch
    auto start1 = high_resolution_clock::now();
    torch::data::samplers::RandomSampler sampler1(dataset_size);
    size_t total1 = 0;
    while (auto batch = sampler1.next(batch_size)) {
        total1 += batch->size();
    }
    auto end1 = high_resolution_clock::now();
    auto elapsed1 = duration_cast<microseconds>(end1 - start1).count();

    // LibTorch-Free
    auto start2 = high_resolution_clock::now();
    lfs::training::RandomSampler sampler2(dataset_size);
    size_t total2 = 0;
    while (auto batch = sampler2.next(batch_size)) {
        total2 += batch->size();
    }
    auto end2 = high_resolution_clock::now();
    auto elapsed2 = duration_cast<microseconds>(end2 - start2).count();

    EXPECT_EQ(total1, dataset_size);
    EXPECT_EQ(total2, dataset_size);

    print_comparison("Large Dataset (1M items)", elapsed1, elapsed2, total1);
}

// ===================================================================================
// Memory Footprint Tests
// ===================================================================================

TEST(DataLoaderMemory, ComponentSizes) {
    std::cout << "\nMemory Footprint:" << std::endl;
    std::cout << "  sizeof(RandomSampler): " << sizeof(RandomSampler) << " bytes" << std::endl;
    std::cout << "  sizeof(InfiniteRandomSampler): " << sizeof(InfiniteRandomSampler) << " bytes" << std::endl;
    std::cout << "  sizeof(ThreadSafeQueue<int>): " << sizeof(ThreadSafeQueue<int>) << " bytes" << std::endl;
    std::cout << "  sizeof(CameraWithImage): " << sizeof(CameraWithImage) << " bytes" << std::endl;
    std::cout << "  sizeof(CameraExample): " << sizeof(CameraExample) << " bytes" << std::endl;
    std::cout << "  sizeof(DataLoaderOptions): " << sizeof(DataLoaderOptions) << " bytes" << std::endl;

    // Verify sizes are reasonable
    EXPECT_LT(sizeof(RandomSampler), 128);  // Should be compact
    EXPECT_LT(sizeof(ThreadSafeQueue<int>), 256);  // Reasonable for thread-safe structure
}
