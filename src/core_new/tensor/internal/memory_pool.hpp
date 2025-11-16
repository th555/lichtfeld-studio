/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/logger.hpp"
#include "gpu_arena_allocator.hpp"
#include "allocation_profiler.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace lfs::core {

    // Threshold for using arena allocator vs cudaMallocAsync
    // Tensors < 100MB use arena (O(1) allocation)
    // Tensors ≥ 100MB use VMM or cudaMallocAsync (avoid arena fragmentation)
    static constexpr size_t ARENA_ALLOCATOR_THRESHOLD = 100 * 1024 * 1024; // 100MB
    static constexpr size_t VMM_ALLOCATOR_THRESHOLD = 100 * 1024 * 1024;   // 100MB - use VMM for ≥100MB (cuMemCreate overhead too high for smaller)
    static constexpr size_t DIRECT_ALLOC_THRESHOLD = 1024 * 1024 * 1024;   // 1GB - bypass pool for very large allocations (fallback)

    /**
     * @brief Allocation method used for a pointer
     *
     * CRITICAL: We MUST track allocation method because deallocation differs:
     * - Arena: Return to arena pool
     * - VMM: Use VMM deallocate (decommits physical, keeps virtual mapping)
     * - Async: Use cudaFreeAsync (returns to CUDA cache for reuse)
     * - Direct: Use cudaFree (immediately returns to OS/driver)
     *
     * Mixing these causes memory leaks!
     */
    enum class AllocMethod : uint8_t {
        Arena, // From GPUArenaAllocator
        Async, // From cudaMallocAsync (CUDA 12.8+)
        Direct // From cudaMalloc (large allocations, fallback)
    };

    /**
     * @brief CUDA memory pool for fast allocation/deallocation
     *
     * Uses cudaMallocAsync with memory pools (CUDA 12.8+) for near-instant
     * allocation from cached memory. Falls back to regular cudaMalloc on older
     * CUDA versions.
     *
     * Performance impact:
     * - cudaMallocAsync from pool: ~0.001-0.01ms (50-600× faster!)
     * - Regular cudaMalloc: ~0.15-0.6ms
     *
     * Expected speedup: 2-10× for typical tensor operations
     */
    class CudaMemoryPool {
    public:
        static CudaMemoryPool& instance() {
            static CudaMemoryPool pool;
            return pool;
        }

        /**
         * @brief Allocate memory from the pool
         * @param bytes Number of bytes to allocate
         * @param stream CUDA stream for stream-ordered allocation
         * @return Pointer to allocated memory, or nullptr on failure
         *
         * Hybrid allocation strategy:
         * - Small/medium (<100MB): Use GPU arena (O(1), 150-600× faster)
         * - Large (≥100MB): Use cudaMallocAsync pool (avoid fragmentation)
         */
        void* allocate(size_t bytes, cudaStream_t stream = nullptr) {
            if (bytes == 0) {
                return nullptr;
            }

            void* ptr = nullptr;

            // ALLOCATION STRATEGY: cudaMallocAsync for all sizes < 1GB, direct cudaMalloc for ≥1GB
            //
            // Why cudaMallocAsync instead of VMM:
            // - cudaMallocAsync: <1 μs per allocation (driver-managed pools, highly optimized)
            // - VMM with cuMemCreate: 100-500 μs per allocation (needs chunk pooling to compete)
            //
            // Memory release strategy:
            // - cudaMallocAsync caches freed memory (fast reuse, but holds onto memory)
            // - We trim pools on allocation failure (cudaMemPoolTrimTo) to release to OS
            // - This gives 95% of VMM benefits without the complexity
            //
            // If you need more aggressive memory release, use trim_memory_pools() manually

            // 1. TODO: Arena for small allocations (<1MB) - still has stream-ordering issues
            // if (bytes < VMM_ALLOCATOR_THRESHOLD && GPUArenaAllocator::instance().is_enabled()) {
            //     ptr = GPUArenaAllocator::instance().allocate(bytes);
            //     if (ptr != nullptr) {
            //         return ptr;
            //     }
            // }

            // 2. DISABLED: VMM for medium/large allocations (too slow without chunk pooling)
            // if (bytes >= VMM_ALLOCATOR_THRESHOLD && bytes < DIRECT_ALLOC_THRESHOLD) {
            //     if (get_or_create_vmm()) {
            //         ptr = vmm_->allocate(bytes, stream);
            //         if (ptr != nullptr) {
            //             std::lock_guard<std::mutex> lock(map_mutex_);
            //             allocation_map_[ptr] = AllocMethod::VMM;
            //             return ptr;
            //         }
            //     }
            // }

            // 3. Direct allocation for very large (≥1GB) or all allocations (VMM disabled)

            if (bytes >= DIRECT_ALLOC_THRESHOLD) {
                // Check available memory before allocation
                size_t free_mem = 0, total_mem = 0;
                cudaMemGetInfo(&free_mem, &total_mem);

                // Log large direct allocations
                static std::atomic<size_t> direct_total{0};
                direct_total += bytes;
                printf("[POOL] Direct cudaMalloc: %.2f GB (total direct: %.2f GB)\n",
                       bytes / (1024.0 * 1024.0 * 1024.0),
                       direct_total.load() / (1024.0 * 1024.0 * 1024.0));

                cudaError_t err = cudaMalloc(&ptr, bytes);
                if (err != cudaSuccess) {
                    printf("[POOL ERROR] cudaMalloc (direct) failed: %s\n", cudaGetErrorString(err));
                    printf("[POOL ERROR] This might be due to fragmentation. Trying to free cached memory...\n");

                    // CRITICAL: Free CUDA memory pool cache to release memory back to OS
                    // This releases memory from cudaMallocAsync pool that's cached but unused
                    cudaDeviceSynchronize(); // Ensure all ops complete

#if CUDART_VERSION >= 12080
                    // Trim the memory pool to release unused memory
                    int device;
                    cudaGetDevice(&device);
                    cudaMemPool_t pool;
                    cudaDeviceGetDefaultMemPool(&pool, device);
                    cudaMemPoolTrimTo(pool, 0); // Release all unused memory
                    printf("[POOL] Trimmed CUDA memory pool\n");
#endif

                    // Check what memory is available after trimming
                    cudaMemGetInfo(&free_mem, &total_mem);
                    printf("[POOL] After trim: free=%.2f GB, requested=%.2f GB\n",
                           free_mem / (1024.0 * 1024.0 * 1024.0),
                           bytes / (1024.0 * 1024.0 * 1024.0));

                    // Retry allocation
                    err = cudaMalloc(&ptr, bytes);
                    if (err != cudaSuccess) {
                        cudaMemGetInfo(&free_mem, &total_mem);
                        printf("[POOL ERROR] Retry failed. Free mem: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
                        LOG_ERROR("cudaMalloc (direct) failed for {} bytes: {}", bytes, cudaGetErrorString(err));
                        LOG_ERROR("Even after trimming pool, not enough memory available.");
                        LOG_ERROR("This indicates intermediate tensors are holding memory.");
                        return nullptr;
                    }
                }

                // CRITICAL: Track that this pointer came from direct cudaMalloc
                // Must use cudaFree (not cudaFreeAsync) for deallocation!
                {
                    std::lock_guard<std::mutex> lock(map_mutex_);
                    allocation_map_[ptr] = AllocMethod::Direct;
                }
                return ptr;
            }

            // Use cudaMallocAsync for medium-large allocations (100MB - 1GB)
            AllocMethod method;
#if CUDART_VERSION >= 12080
            // Use stream-ordered allocation with memory pool (FAST!)
            cudaError_t err = cudaMallocAsync(&ptr, bytes, stream);
            if (err == cudaSuccess) {
                method = AllocMethod::Async;

                // Record allocation site for profiling
                AllocationProfiler::instance().record_allocation(bytes, 3); // skip 3 frames: record_allocation, allocate, Tensor::allocate

                // Track allocation sizes
                static std::atomic<int> alloc_count{0};
                static std::atomic<size_t> total_bytes_allocated{0};
                static std::atomic<int> small_allocs{0};   // < 1 MB
                static std::atomic<int> medium_allocs{0};  // 1-100 MB
                static std::atomic<int> large_allocs{0};   // 100MB-1GB

                alloc_count++;
                total_bytes_allocated += bytes;

                if (bytes < (1 << 20)) small_allocs++;
                else if (bytes < (100 << 20)) medium_allocs++;
                else large_allocs++;

                if (alloc_count % 2000 == 0) {
                    // Print allocation profiling report every 2k allocations
                    if constexpr (ENABLE_ALLOCATION_PROFILING) {
                        AllocationProfiler::instance().print_top_allocators(30);
                    }

                    // Query memory pool attributes
                    int device;
                    cudaGetDevice(&device);
                    cudaMemPool_t pool;
                    cudaDeviceGetDefaultMemPool(&pool, device);

                    // Get pool reserved memory (what's actually allocated from OS)
                    uint64_t pool_reserved = 0;
                    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &pool_reserved);

                    // Get pool used memory (what's actively in use)
                    uint64_t pool_used = 0;
                    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &pool_used);

                    // Get total process CUDA memory usage
                    size_t free_mem, total_mem;
                    cudaMemGetInfo(&free_mem, &total_mem);
                    size_t process_used = total_mem - free_mem;

                    // Calculate non-pool memory (direct allocations + arena + driver overhead)
                    size_t non_pool = process_used - pool_reserved;

                    printf("[POOL] After %d allocs (small:%d med:%d large:%d) | Pool: reserved=%.2f GB, used=%.2f GB | Cumulative allocated: %.2f GB | Process: %.2f GB (non-pool: %.2f GB)\n",
                           alloc_count.load(),
                           small_allocs.load(),
                           medium_allocs.load(),
                           large_allocs.load(),
                           pool_reserved / (1024.0 * 1024.0 * 1024.0),
                           pool_used / (1024.0 * 1024.0 * 1024.0),
                           total_bytes_allocated.load() / (1024.0 * 1024.0 * 1024.0),
                           process_used / (1024.0 * 1024.0 * 1024.0),
                           non_pool / (1024.0 * 1024.0 * 1024.0));
                }
            } else {
                LOG_ERROR("cudaMallocAsync failed for {} bytes: {}",
                          bytes, cudaGetErrorString(err));
                // Fallback to direct allocation if pool fails
                LOG_WARN("Falling back to direct cudaMalloc");
                err = cudaMalloc(&ptr, bytes);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMalloc (fallback) failed for {} bytes: {}",
                              bytes, cudaGetErrorString(err));
                    return nullptr;
                }
                method = AllocMethod::Direct; // Fallback uses direct
            }
#else
            // Fallback to synchronous allocation (SLOW) for CUDA < 12.8
            cudaError_t err = cudaMalloc(&ptr, bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaMalloc failed for {} bytes: {}",
                          bytes, cudaGetErrorString(err));
                return nullptr;
            }
            method = AllocMethod::Direct; // Old CUDA version uses direct
            LOG_WARN("Using cudaMalloc (CUDA < 12.8). Consider upgrading for 50-600× faster allocation");
#endif

            // Track allocation method (not Arena, so must track in map)
            {
                std::lock_guard<std::mutex> lock(map_mutex_);
                allocation_map_[ptr] = method;
            }

            return ptr;
        }

        /**
         * @brief Deallocate memory back to the pool
         * @param ptr Pointer to memory to deallocate
         * @param stream CUDA stream for stream-ordered deallocation
         *
         * CRITICAL: Uses correct deallocation method based on how pointer was allocated:
         * - Arena → Return to arena pool
         * - Async (cudaMallocAsync) → cudaFreeAsync (returns to CUDA cache)
         * - Direct (cudaMalloc) → cudaFree (returns to OS/driver)
         *
         * Mixing allocation/deallocation methods causes memory leaks!
         */
        void deallocate(void* ptr, cudaStream_t stream = nullptr) {
            if (!ptr) {
                return;
            }

            // DISABLED: Arena allocator has stream-ordering bugs
            // if (GPUArenaAllocator::instance().is_enabled() &&
            //     GPUArenaAllocator::instance().owns_pointer(ptr)) {
            //     GPUArenaAllocator::instance().deallocate(ptr);
            //     return;
            // }

            // Look up allocation method from tracking map
            AllocMethod method;
            {
                std::lock_guard<std::mutex> lock(map_mutex_);
                auto it = allocation_map_.find(ptr);
                if (it == allocation_map_.end()) {
                    LOG_ERROR("Attempting to deallocate untracked pointer: {}", ptr);
                    LOG_ERROR("This likely means the pointer was allocated before tracking was added,");
                    LOG_ERROR("or there's a double-free bug. Attempting cudaFree as fallback.");
                    cudaFree(ptr); // Best-effort fallback
                    return;
                }
                method = it->second;
                allocation_map_.erase(it); // Remove from tracking
            }

            // CRITICAL: Use correct deallocation function based on allocation method!
            cudaError_t err;
            switch (method) {
            case AllocMethod::Direct:
                // Direct cudaMalloc requires cudaFree (synchronous, returns to OS)
                err = cudaFree(ptr);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaFree failed for Direct allocation: {}", cudaGetErrorString(err));
                }
                break;

            case AllocMethod::Async:
                // cudaMallocAsync requires cudaFreeAsync (returns to CUDA cache)
#if CUDART_VERSION >= 12080
                err = cudaFreeAsync(ptr, stream);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaFreeAsync failed for Async allocation: {}", cudaGetErrorString(err));
                }

                // No logging for deallocations - too noisy
#else
                // Shouldn't happen (Async not used on old CUDA), but fallback to cudaFree
                LOG_WARN("Unexpected Async allocation on CUDA < 12.8, using cudaFree");
                err = cudaFree(ptr);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaFree failed: {}", cudaGetErrorString(err));
                }
#endif
                break;

            case AllocMethod::Arena:
                // Should have been caught by owns_pointer check above
                LOG_ERROR("Arena allocation not caught by owns_pointer check!");
                GPUArenaAllocator::instance().deallocate(ptr);
                break;

            default:
                LOG_ERROR("Unknown allocation method for pointer: {}", ptr);
                break;
            }
        }

        /**
         * @brief Configure memory pool settings for optimal performance
         */
        void configure() {
#if CUDART_VERSION >= 12080
            int device;
            cudaError_t err = cudaGetDevice(&device);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaGetDevice failed: {}", cudaGetErrorString(err));
                return;
            }

            // Get the default memory pool for this device
            cudaMemPool_t pool;
            err = cudaDeviceGetDefaultMemPool(&pool, device);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaDeviceGetDefaultMemPool failed: {}", cudaGetErrorString(err));
                return;
            }

            // Set pool release threshold - keep memory cached indefinitely
            // This prevents the pool from releasing memory back to the system,
            // maximizing reuse and minimizing allocation overhead
            uint64_t threshold = UINT64_MAX;
            err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            if (err != cudaSuccess) {
                LOG_WARN("cudaMemPoolSetAttribute failed: {}", cudaGetErrorString(err));
            }

            LOG_INFO("CUDA memory pool configured for device {} (CUDA {})",
                     device, CUDART_VERSION);
            LOG_INFO("Memory pool will cache allocations for maximum performance");
#else
            LOG_WARN("CUDA memory pooling not available (requires CUDA >= 12.8, current: {})",
                     CUDART_VERSION);
            LOG_WARN("Performance will be 50-600× slower than with memory pooling");
#endif
        }

        /**
         * @brief Get statistics about the memory pool
         * @return String with pool statistics (empty on CUDA < 12.8)
         */
        std::string get_stats() const {
#if CUDART_VERSION >= 12080
            int device;
            cudaGetDevice(&device);

            cudaMemPool_t pool;
            cudaDeviceGetDefaultMemPool(&pool, device);

            // Get used memory
            uint64_t used_memory = 0;
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_memory);

            // Get reserved memory
            uint64_t reserved_memory = 0;
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved_memory);

            std::ostringstream oss;
            oss << "Memory Pool Stats:\n";
            oss << "  Used:     " << (used_memory / 1024.0 / 1024.0) << " MB\n";
            oss << "  Reserved: " << (reserved_memory / 1024.0 / 1024.0) << " MB\n";
            oss << "  Cached:   " << ((reserved_memory - used_memory) / 1024.0 / 1024.0) << " MB";
            return oss.str();
#else
            return "Memory pool statistics not available (CUDA < 12.8)";
#endif
        }

        /**
         * @brief Trim the memory pool, releasing unused memory back to the system
         *
         * This can be called periodically if memory pressure is high, but generally
         * it's better to keep memory cached for performance.
         */
        void trim() {
#if CUDART_VERSION >= 12080
            int device;
            cudaGetDevice(&device);

            cudaMemPool_t pool;
            cudaDeviceGetDefaultMemPool(&pool, device);

            // Release memory above the threshold
            cudaError_t err = cudaMemPoolTrimTo(pool, 0);
            if (err != cudaSuccess) {
                LOG_WARN("cudaMemPoolTrimTo failed: {}", cudaGetErrorString(err));
            } else {
                LOG_INFO("Memory pool trimmed successfully");
            }
#else
            LOG_DEBUG("Memory pool trim not available (CUDA < 12.8)");
#endif
        }

        /**
         * @brief Manually release cached memory back to OS
         *
         * Call this between training runs or when you need to free memory for other processes.
         * cudaMallocAsync caches freed memory for fast reuse, but this releases it to the OS.
         *
         * Example usage:
         *   CudaMemoryPool::instance().trim_cached_memory();  // After densification
         *   CudaMemoryPool::instance().trim_cached_memory();  // Between training runs
         */
        void trim_cached_memory() {
#if CUDART_VERSION >= 12080
            cudaDeviceSynchronize(); // Ensure all operations complete

            int device;
            cudaGetDevice(&device);
            cudaMemPool_t pool;
            if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
                size_t before_free = 0, total = 0;
                cudaMemGetInfo(&before_free, &total);

                cudaMemPoolTrimTo(pool, 0); // Release all unused cached memory

                size_t after_free = 0;
                cudaMemGetInfo(&after_free, &total);

                // Removed verbose logging - memory pool trimming is expected behavior
                // if (after_free > before_free) {
                //     printf("[MEMORY] Trimmed pool: freed %.2f GB\n",
                //            (after_free - before_free) / (1024.0 * 1024.0 * 1024.0));
                // }
            }
#endif
        }

        // Disable copy and move
        CudaMemoryPool(const CudaMemoryPool&) = delete;
        CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;
        CudaMemoryPool(CudaMemoryPool&&) = delete;
        CudaMemoryPool& operator=(CudaMemoryPool&&) = delete;

    private:
        CudaMemoryPool() {
            configure();
        }

        ~CudaMemoryPool() {
            // Memory pool is automatically cleaned up by CUDA runtime
        }

        // Thread-safe tracking of allocation methods
        // CRITICAL: We must know which free function to call for each pointer!
        std::unordered_map<void*, AllocMethod> allocation_map_;
        std::mutex map_mutex_;
    };

} // namespace lfs::core
