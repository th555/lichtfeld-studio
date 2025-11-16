/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core_new/pinned_memory_allocator.hpp"
#include "core_new/tensor/internal/memory_pool.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>

// Custom event listener to clean GPU memory before AND after each test
class MemoryCleanupListener : public ::testing::EmptyTestEventListener {
private:
    void aggressive_cleanup() {
        cudaSetDevice(0);

        // Synchronize and check for errors
        cudaError_t sync_err = cudaDeviceSynchronize();
        cudaError_t last_err = cudaGetLastError();

        // If there are CUDA errors, log and clear them
        if (sync_err != cudaSuccess || last_err != cudaSuccess) {
            if (sync_err != cudaSuccess) {
                std::cerr << "[CLEANUP] CUDA sync error: " << cudaGetErrorString(sync_err) << std::endl;
            }
            if (last_err != cudaSuccess) {
                std::cerr << "[CLEANUP] CUDA error: " << cudaGetErrorString(last_err) << std::endl;
            }
            // Clear the error
            cudaGetLastError();
        }

        // Empty PyTorch's cache
        try {
            c10::cuda::CUDACachingAllocator::emptyCache();
        } catch (...) {
            std::cerr << "[CLEANUP] Exception while emptying PyTorch cache" << std::endl;
        }

        // Trim custom CUDA memory pool
        try {
            lfs::core::CudaMemoryPool::instance().trim();
        } catch (...) {
            std::cerr << "[CLEANUP] Exception while trimming custom pool" << std::endl;
        }

        // Final sync
        cudaDeviceSynchronize();
        cudaGetLastError();  // Clear any remaining errors
    }

public:
    void OnTestStart(const ::testing::TestInfo& /*test_info*/) override {
        aggressive_cleanup();
    }

    void OnTestEnd(const ::testing::TestInfo& /*test_info*/) override {
        aggressive_cleanup();
    }
};

int main(int argc, char** argv) {
    // Initialize logger with Info level
    gs::core::Logger::get().init(gs::core::LogLevel::Info);

    ::testing::InitGoogleTest(&argc, argv);

    // Add custom listener to clean memory before each test
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new MemoryCleanupListener);

    // Pre-warm pinned memory cache for fast CPU-GPU transfers
    // This eliminates cold-start penalties (e.g., 23.8ms for 4K allocations)
    lfs::core::PinnedMemoryAllocator::instance().prewarm();

    return RUN_ALL_TESTS();
}
