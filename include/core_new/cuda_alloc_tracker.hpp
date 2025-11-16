/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <string>
#include <cstdio>

#ifdef __linux__
#include <execinfo.h>
#include <cxxabi.h>
#endif

namespace lfs::core {

class CudaAllocTracker {
public:
    static CudaAllocTracker& instance() {
        static CudaAllocTracker tracker;
        return tracker;
    }

    void record_alloc(void* ptr, size_t bytes, const char* location) {
        if (!ptr) return;

        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[ptr] = {bytes, location ? location : "unknown"};
        total_allocated_ += bytes;

        if (total_allocated_ / (1024*1024*1024.0) > last_print_gb_ + 0.5) {
            last_print_gb_ = total_allocated_ / (1024*1024*1024.0);
            print_summary();
        }
    }

    void record_free(void* ptr) {
        if (!ptr) return;

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            total_freed_ += it->second.bytes;
            allocations_.erase(it);
        }
    }

    void print_summary() {
        std::lock_guard<std::mutex> lock(mutex_);

        printf("\n========== CUDA ALLOCATION TRACKER ==========\n");
        printf("Total allocated: %.2f GB\n", total_allocated_ / (1024.0*1024*1024));
        printf("Total freed: %.2f GB\n", total_freed_ / (1024.0*1024*1024));
        printf("Currently allocated: %.2f GB (%zu allocations)\n",
               (total_allocated_ - total_freed_) / (1024.0*1024*1024),
               allocations_.size());

        // Group by location
        std::unordered_map<std::string, size_t> by_location;
        for (const auto& [ptr, info] : allocations_) {
            by_location[info.location] += info.bytes;
        }

        printf("\nTop allocations by location:\n");
        std::vector<std::pair<std::string, size_t>> sorted(by_location.begin(), by_location.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        for (size_t i = 0; i < std::min(size_t(10), sorted.size()); i++) {
            printf("  %60s: %.2f MB\n",
                   sorted[i].first.c_str(),
                   sorted[i].second / (1024.0*1024));
        }
        printf("=============================================\n\n");
    }

private:
    struct AllocInfo {
        size_t bytes;
        std::string location;
    };

    CudaAllocTracker() = default;

    std::mutex mutex_;
    std::unordered_map<void*, AllocInfo> allocations_;
    size_t total_allocated_ = 0;
    size_t total_freed_ = 0;
    double last_print_gb_ = 0;
};

// Wrapper functions to track cudaMalloc/cudaFree
#define TRACKED_CUDA_MALLOC(ptr, size, location) \
    do { \
        cudaError_t err = cudaMalloc(ptr, size); \
        if (err == cudaSuccess) { \
            lfs::core::CudaAllocTracker::instance().record_alloc(*(ptr), size, location); \
        } \
    } while(0)

#define TRACKED_CUDA_FREE(ptr) \
    do { \
        lfs::core::CudaAllocTracker::instance().record_free(ptr); \
        cudaFree(ptr); \
    } while(0)

} // namespace lfs::core
