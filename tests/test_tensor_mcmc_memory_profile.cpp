/* Test MCMC-style memory usage with large tensors and profiling */

#include <gtest/gtest.h>
#include "core_new/tensor.hpp"
#include <cuda_runtime.h>

using namespace lfs::core;

namespace {

// Helper to get GPU memory info
struct MemoryInfo {
    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes() const { return total_bytes - free_bytes; }
    double used_gb() const { return used_bytes() / (1024.0 * 1024.0 * 1024.0); }
    double free_gb() const { return free_bytes / (1024.0 * 1024.0 * 1024.0); }
};

MemoryInfo get_memory_info() {
    MemoryInfo info;
    cudaMemGetInfo(&info.free_bytes, &info.total_bytes);
    return info;
}

} // anonymous namespace

TEST(TensorMCMCMemoryProfile, LargeVectorReserveAndGrow) {
    // Mimic MCMC: Start with a reasonable size, reserve for max capacity, then grow
    printf("\n=== MCMC-style Memory Profile Test ===\n");

    auto mem_start = get_memory_info();
    printf("Initial: %.2f GB used, %.2f GB free\n", mem_start.used_gb(), mem_start.free_gb());

    // Start with 100k Gaussians
    const size_t initial_size = 100000;
    const size_t max_capacity = 4000000;  // 4M max capacity like in MCMC

    printf("\nCreating tensors for %zu initial Gaussians (reserving for %zu)...\n",
           initial_size, max_capacity);

    // Create typical MCMC tensors
    Tensor means = Tensor::zeros({initial_size, 3}, Device::CUDA, DataType::Float32);
    Tensor scales = Tensor::ones({initial_size, 3}, Device::CUDA, DataType::Float32);
    Tensor rotations = Tensor::zeros({initial_size, 4}, Device::CUDA, DataType::Float32);
    Tensor opacities = Tensor::ones({initial_size, 1}, Device::CUDA, DataType::Float32);
    Tensor features = Tensor::zeros({initial_size, 15, 3}, Device::CUDA, DataType::Float32);  // SH features

    auto mem_after_create = get_memory_info();
    printf("After creation: %.2f GB used (+%.2f GB), %.2f GB free\n",
           mem_after_create.used_gb(),
           mem_after_create.used_gb() - mem_start.used_gb(),
           mem_after_create.free_gb());

    // Reserve capacity for all tensors
    // CRITICAL: reserve() takes NUMBER OF ROWS (first dimension), not total elements!
    // The Tensor class automatically calculates bytes based on the full shape.
    printf("\nReserving capacity for %zu Gaussians...\n", max_capacity);
    means.reserve(max_capacity);
    scales.reserve(max_capacity);
    rotations.reserve(max_capacity);
    opacities.reserve(max_capacity);
    features.reserve(max_capacity);

    auto mem_after_reserve = get_memory_info();
    printf("After reserve: %.2f GB used (+%.2f GB), %.2f GB free\n",
           mem_after_reserve.used_gb(),
           mem_after_reserve.used_gb() - mem_after_create.used_gb(),
           mem_after_reserve.free_gb());

    // Simulate MCMC growth: add 500k Gaussians in chunks
    printf("\nSimulating MCMC growth by adding 500k Gaussians in 10k chunks...\n");
    const size_t chunk_size = 10000;
    const size_t num_chunks = 50;  // 50 * 10k = 500k

    size_t current_size = initial_size;
    for (size_t i = 0; i < num_chunks; i++) {
        // Create new Gaussians
        Tensor new_means = Tensor::ones({chunk_size, 3}, Device::CUDA, DataType::Float32);
        Tensor new_scales = Tensor::ones({chunk_size, 3}, Device::CUDA, DataType::Float32);
        Tensor new_rotations = Tensor::zeros({chunk_size, 4}, Device::CUDA, DataType::Float32);
        Tensor new_opacities = Tensor::ones({chunk_size, 1}, Device::CUDA, DataType::Float32);
        Tensor new_features = Tensor::zeros({chunk_size, 15, 3}, Device::CUDA, DataType::Float32);

        // Append to existing tensors (this is what MCMC does)
        means = means.cat({new_means}, 0);
        scales = scales.cat({new_scales}, 0);
        rotations = rotations.cat({new_rotations}, 0);
        opacities = opacities.cat({new_opacities}, 0);
        features = features.cat({new_features}, 0);

        current_size += chunk_size;

        // Print memory every 10 chunks
        if ((i + 1) % 10 == 0) {
            auto mem_now = get_memory_info();
            printf("  After %zu Gaussians: %.2f GB used, %.2f GB free\n",
                   current_size, mem_now.used_gb(), mem_now.free_gb());
        }
    }

    auto mem_after_growth = get_memory_info();
    printf("\nAfter growing to %zu Gaussians:\n", current_size);
    printf("  Memory used: %.2f GB (+%.2f GB from reserve)\n",
           mem_after_growth.used_gb(),
           mem_after_growth.used_gb() - mem_after_reserve.used_gb());
    printf("  Memory free: %.2f GB\n", mem_after_growth.free_gb());

    // Verify shapes
    EXPECT_EQ(means.shape()[0], 600000);
    EXPECT_EQ(scales.shape()[0], 600000);
    EXPECT_EQ(rotations.shape()[0], 600000);
    EXPECT_EQ(opacities.shape()[0], 600000);
    EXPECT_EQ(features.shape()[0], 600000);

    printf("\n=== Test Complete ===\n");
}

TEST(TensorMCMCMemoryProfile, MassiveGrowthTo1M) {
    // Even more aggressive: grow from 100k to 1M
    printf("\n=== Massive Growth Test (100k -> 1M) ===\n");

    auto mem_start = get_memory_info();
    printf("Initial: %.2f GB used, %.2f GB free\n", mem_start.used_gb(), mem_start.free_gb());

    const size_t initial_size = 100000;
    const size_t max_capacity = 4000000;

    // Create and reserve for spherical harmonics features (largest tensor)
    printf("\nCreating SH features [%zu, 15, 3] and reserving for %zu...\n",
           initial_size, max_capacity);
    Tensor features = Tensor::zeros({initial_size, 15, 3}, Device::CUDA, DataType::Float32);

    auto mem_after_create = get_memory_info();
    printf("After creation: %.2f GB used, %.2f GB free\n",
           mem_after_create.used_gb(), mem_after_create.free_gb());

    features.reserve(max_capacity);

    auto mem_after_reserve = get_memory_info();
    printf("After reserve: %.2f GB used (+%.2f GB), %.2f GB free\n",
           mem_after_reserve.used_gb(),
           mem_after_reserve.used_gb() - mem_after_create.used_gb(),
           mem_after_reserve.free_gb());

    // Grow to 1M in large chunks
    printf("\nGrowing to 1M in 50k chunks...\n");
    const size_t chunk_size = 50000;
    size_t current_size = initial_size;

    while (current_size < 1000000) {
        Tensor new_features = Tensor::ones({chunk_size, 15, 3}, Device::CUDA, DataType::Float32);
        features = features.cat({new_features}, 0);
        current_size += chunk_size;

        auto mem_now = get_memory_info();
        printf("  After %zu: %.2f GB used, %.2f GB free\n",
               current_size, mem_now.used_gb(), mem_now.free_gb());
    }

    auto mem_final = get_memory_info();
    printf("\nFinal state with %zu Gaussians:\n", current_size);
    printf("  Memory used: %.2f GB\n", mem_final.used_gb());
    printf("  Memory free: %.2f GB\n", mem_final.free_gb());
    printf("  Memory overhead from reserve: %.2f GB\n",
           mem_final.used_gb() - mem_after_reserve.used_gb());

    EXPECT_EQ(features.shape()[0], 1000000);
    EXPECT_EQ(features.shape()[1], 15);
    EXPECT_EQ(features.shape()[2], 3);

    printf("\n=== Test Complete ===\n");
}

TEST(TensorMCMCMemoryProfile, CompareReserveVsNoReserve) {
    // Compare memory behavior with and without reserve
    printf("\n=== Reserve vs No-Reserve Comparison ===\n");

    const size_t initial_size = 50000;
    const size_t final_size = 500000;
    const size_t chunk_size = 50000;

    // Test WITH reserve
    {
        printf("\n--- WITH reserve() ---\n");
        auto mem_start = get_memory_info();
        printf("Start: %.2f GB used\n", mem_start.used_gb());

        Tensor t = Tensor::zeros({initial_size, 15, 3}, Device::CUDA, DataType::Float32);
        t.reserve(4000000);

        auto mem_after_reserve = get_memory_info();
        printf("After reserve: %.2f GB used (+%.2f GB)\n",
               mem_after_reserve.used_gb(),
               mem_after_reserve.used_gb() - mem_start.used_gb());

        size_t current_size = initial_size;
        while (current_size < final_size) {
            Tensor new_data = Tensor::ones({chunk_size, 15, 3}, Device::CUDA, DataType::Float32);
            t = t.cat({new_data}, 0);
            current_size += chunk_size;
        }

        auto mem_final = get_memory_info();
        printf("After growth to %zu: %.2f GB used (+%.2f GB from reserve)\n",
               current_size, mem_final.used_gb(),
               mem_final.used_gb() - mem_after_reserve.used_gb());

        EXPECT_EQ(t.shape()[0], 500000);
    }

    // Test WITHOUT reserve
    {
        printf("\n--- WITHOUT reserve() ---\n");
        auto mem_start = get_memory_info();
        printf("Start: %.2f GB used\n", mem_start.used_gb());

        Tensor t = Tensor::zeros({initial_size, 15, 3}, Device::CUDA, DataType::Float32);

        auto mem_after_create = get_memory_info();
        printf("After creation: %.2f GB used\n", mem_after_create.used_gb());

        size_t current_size = initial_size;
        while (current_size < final_size) {
            Tensor new_data = Tensor::ones({chunk_size, 15, 3}, Device::CUDA, DataType::Float32);
            t = t.cat({new_data}, 0);
            current_size += chunk_size;
        }

        auto mem_final = get_memory_info();
        printf("After growth to %zu: %.2f GB used (+%.2f GB)\n",
               current_size, mem_final.used_gb(),
               mem_final.used_gb() - mem_after_create.used_gb());

        EXPECT_EQ(t.shape()[0], 500000);
    }

    printf("\n=== Comparison Complete ===\n");
}
