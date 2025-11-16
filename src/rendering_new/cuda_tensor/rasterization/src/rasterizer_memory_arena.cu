/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rasterizer_memory_arena.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace lfs::rendering {

    // Default constructor implementation
    RasterizerMemoryArena::RasterizerMemoryArena()
        : RasterizerMemoryArena(Config{}) {
    }

    RasterizerMemoryArena::RasterizerMemoryArena(const Config& cfg)
        : config_(cfg),
          creation_time_(std::chrono::steady_clock::now()) {

        // Check if VMM is supported
        int device;
        cudaGetDevice(&device);
        if (!is_vmm_supported(device)) {
            std::cerr << "[RasterizerMemoryArena] WARNING: VMM not supported, falling back to smaller allocation\n";
            config_.initial_commit = 128 << 20; // Smaller initial if no VMM
            config_.max_physical = 4ULL << 30;  // Lower max if no VMM
        }

        std::cout << "[RasterizerMemoryArena] Created with config:\n"
                  << "  Virtual size: " << (config_.virtual_size >> 30) << " GB (costs no memory!)\n"
                  << "  Initial commit: " << (config_.initial_commit >> 20) << " MB\n"
                  << "  Max physical: " << (config_.max_physical >> 30) << " GB\n"
                  << "  Granularity: " << (config_.granularity >> 20) << " MB\n"
                  << "  Alignment: " << config_.alignment << " bytes\n"
                  << "  Log interval: every " << config_.log_interval << " frames\n";
    }

    RasterizerMemoryArena::~RasterizerMemoryArena() {
        dump_statistics();

        // Clean up all arenas
        std::lock_guard<std::mutex> lock(arena_mutex_);
        for (auto& [device, arena_ptr] : device_arenas_) {
            if (!arena_ptr)
                continue;
            auto& arena = *arena_ptr;

            // Unmap and release all physical memory
            std::lock_guard<std::mutex> chunk_lock(arena.chunks_mutex);
            for (auto& chunk : arena.chunks) {
                if (chunk.is_mapped) {
                    cuMemUnmap(arena.d_ptr + chunk.offset, chunk.size);
                    cuMemRelease(chunk.handle);
                }
            }

            // Free virtual address space
            if (arena.d_ptr) {
                cuMemAddressFree(arena.d_ptr, arena.virtual_size);
            }

            // Free fallback buffer if exists
            if (arena.fallback_buffer) {
                cudaFree(arena.fallback_buffer);
            }
        }

        device_arenas_.clear();
        frame_contexts_.clear();
    }

    RasterizerMemoryArena::RasterizerMemoryArena(RasterizerMemoryArena&& other) noexcept
        : device_arenas_(std::move(other.device_arenas_)),
          frame_contexts_(std::move(other.frame_contexts_)),
          config_(other.config_),
          frame_counter_(other.frame_counter_.load()),
          generation_counter_(other.generation_counter_.load()),
          creation_time_(other.creation_time_),
          total_frames_processed_(other.total_frames_processed_.load()) {
    }

    RasterizerMemoryArena& RasterizerMemoryArena::operator=(RasterizerMemoryArena&& other) noexcept {
        if (this != &other) {
            std::lock_guard<std::mutex> lock1(arena_mutex_);
            std::lock_guard<std::mutex> lock2(other.arena_mutex_);

            device_arenas_ = std::move(other.device_arenas_);
            frame_contexts_ = std::move(other.frame_contexts_);
            config_ = other.config_;
            frame_counter_ = other.frame_counter_.load();
            generation_counter_ = other.generation_counter_.load();
            creation_time_ = other.creation_time_;
            total_frames_processed_ = other.total_frames_processed_.load();
        }
        return *this;
    }

    bool RasterizerMemoryArena::is_vmm_supported(int device) const {
        // Check compute capability
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

        // VMM requires compute capability 6.0+
        if (major < 6) {
            return false;
        }

        // Check if we can get allocation granularity (VMM function)
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;

        size_t granularity = 0;
        CUresult result = cuMemGetAllocationGranularity(&granularity, &prop,
                                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        // If this VMM function succeeds, VMM is supported
        if (result == CUDA_SUCCESS && granularity > 0) {
            return true;
        }

        // VMM not supported or not available
        return false;
    }

    uint64_t RasterizerMemoryArena::begin_frame() {
        uint64_t frame_id = frame_counter_.fetch_add(1, std::memory_order_relaxed);

        // CRITICAL FIX: Reset arena offset at the beginning of each frame!
        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err == cudaSuccess) {
            std::lock_guard<std::mutex> lock(arena_mutex_);
            auto it = device_arenas_.find(device);
            if (it != device_arenas_.end() && it->second) {
                // Reset the offset to reuse the buffer from the beginning
                it->second->offset.store(0, std::memory_order_release);

                // Log memory status periodically (but not too often)
                bool should_log = (frame_id == 1) || (frame_id % config_.log_interval == 0);

                if (should_log) {
                    log_memory_status(frame_id, true);
                }
            }
        }

        std::lock_guard<std::mutex> lock(frame_mutex_);

        // Create new frame context
        FrameContext& ctx = frame_contexts_[frame_id];
        ctx.frame_id = frame_id;
        ctx.generation = generation_counter_.load(std::memory_order_relaxed);
        ctx.is_active = true;
        ctx.timestamp = std::chrono::steady_clock::now();
        ctx.buffers.clear();
        ctx.total_allocated = 0;

        total_frames_processed_.fetch_add(1, std::memory_order_relaxed);

        return frame_id;
    }

    void RasterizerMemoryArena::end_frame(uint64_t frame_id) {
        // Track peak usage before resetting
        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err == cudaSuccess) {
            std::lock_guard<std::mutex> lock(arena_mutex_);
            auto it = device_arenas_.find(device);
            if (it != device_arenas_.end() && it->second) {
                size_t frame_usage = it->second->offset.load(std::memory_order_relaxed);

                // Update overall peak
                size_t current_peak = it->second->peak_usage.load(std::memory_order_relaxed);
                while (frame_usage > current_peak) {
                    if (it->second->peak_usage.compare_exchange_weak(current_peak, frame_usage)) {
                        break;
                    }
                }

                // Update period peak (for logging)
                size_t period_peak = it->second->peak_usage_period.load(std::memory_order_relaxed);
                while (frame_usage > period_peak) {
                    if (it->second->peak_usage_period.compare_exchange_weak(period_peak, frame_usage)) {
                        break;
                    }
                }

                // Decommit unused memory if under pressure
                if (get_memory_pressure() > 0.75f) {
                    decommit_unused_memory(*it->second);
                }
            }
        }

        std::lock_guard<std::mutex> lock(frame_mutex_);

        auto it = frame_contexts_.find(frame_id);
        if (it != frame_contexts_.end()) {
            it->second.is_active = false;
        }

        // Cleanup old frames - keep only last 3
        cleanup_frames(3);
    }

    void RasterizerMemoryArena::log_memory_status(uint64_t frame_id, bool force) {
        // Called with arena_mutex_ already held
        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess)
            return;

        auto it = device_arenas_.find(device);
        if (it == device_arenas_.end() || !it->second)
            return;

        auto& arena = *it->second;

        // Get memory info
        size_t committed_mb = arena.committed_size >> 20;
        size_t current_usage_mb = arena.offset.load() >> 20;
        size_t peak_period_mb = arena.peak_usage_period.load() >> 20;
        size_t peak_overall_mb = arena.peak_usage.load() >> 20;

        // Get GPU memory info
        size_t free_gpu, total_gpu;
        cudaMemGetInfo(&free_gpu, &total_gpu);

        // Calculate utilization
        float utilization = arena.committed_size > 0 ? (100.0f * arena.peak_usage_period.load() / arena.committed_size) : 0.0f;

        // Log the status
        std::cout << "\n[Arena Memory Status] Frame " << frame_id << " | Device " << device << ":\n";

        if (arena.d_ptr != 0) {
            std::cout << "  Virtual reserved: " << (arena.virtual_size >> 30) << " GB (no cost)\n"
                      << "  Physical committed: " << committed_mb << " MB (actual memory)\n";
        } else {
            std::cout << "  Allocated: " << committed_mb << " MB (traditional mode)\n";
        }

        std::cout << "  Peak usage (last " << config_.log_interval << " frames): "
                  << peak_period_mb << " MB (" << std::fixed << std::setprecision(1)
                  << utilization << "%)\n"
                  << "  Peak usage (overall): " << peak_overall_mb << " MB\n"
                  << "  Reallocations: " << arena.realloc_count.load() << "\n"
                  << "  GPU: " << (free_gpu >> 20) << "/" << (total_gpu >> 20)
                  << " MB free (" << std::fixed << std::setprecision(1)
                  << (100.0f * free_gpu / total_gpu) << "% available)\n";

        // Reset period peak for next logging interval
        arena.peak_usage_period.store(0, std::memory_order_release);
        arena.last_log_time = std::chrono::steady_clock::now();
    }

    std::function<char*(size_t)> RasterizerMemoryArena::get_allocator(uint64_t frame_id) {
        return [this, frame_id](size_t size) -> char* {
            if (size == 0) {
                return nullptr;
            }

            int device;
            cudaError_t err = cudaGetDevice(&device);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to get CUDA device: " +
                                         std::string(cudaGetErrorString(err)));
            }

            Arena& arena = get_or_create_arena(device);
            return allocate_internal(arena, size, frame_id);
        };
    }

    std::vector<RasterizerMemoryArena::BufferHandle>
    RasterizerMemoryArena::get_frame_buffers(uint64_t frame_id) const {
        std::lock_guard<std::mutex> lock(frame_mutex_);

        auto it = frame_contexts_.find(frame_id);
        if (it != frame_contexts_.end()) {
            return it->second.buffers;
        }

        return {};
    }

    void RasterizerMemoryArena::reset_frame(uint64_t frame_id) {
        std::lock_guard<std::mutex> lock(frame_mutex_);

        auto it = frame_contexts_.find(frame_id);
        if (it != frame_contexts_.end()) {
            // Keep buffers but mark as reusable
            it->second.total_allocated = 0;
        }
    }

    void RasterizerMemoryArena::cleanup_frames(int keep_recent) {
        // Called with frame_mutex_ already held

        if (frame_contexts_.size() <= static_cast<size_t>(keep_recent)) {
            return;
        }

        // Find oldest frames to remove
        std::vector<uint64_t> frame_ids;
        frame_ids.reserve(frame_contexts_.size());

        for (const auto& [id, ctx] : frame_contexts_) {
            if (!ctx.is_active) {
                frame_ids.push_back(id);
            }
        }

        if (frame_ids.size() <= static_cast<size_t>(keep_recent)) {
            return;
        }

        // Sort by frame ID (oldest first)
        std::sort(frame_ids.begin(), frame_ids.end());

        // Remove oldest frames
        size_t to_remove = frame_ids.size() - keep_recent;
        for (size_t i = 0; i < to_remove; ++i) {
            frame_contexts_.erase(frame_ids[i]);
        }
    }

    void RasterizerMemoryArena::empty_cuda_cache() {
        // Try to force CUDA to release cached memory
        // This is less aggressive than cudaDeviceReset()
        cudaDeviceSynchronize();

        // Allocate and immediately free a small buffer to trigger cleanup
        void* dummy;
        cudaMalloc(&dummy, 1);
        cudaFree(dummy);
    }

    void RasterizerMemoryArena::emergency_cleanup() {
        std::lock_guard<std::mutex> lock1(arena_mutex_);
        std::lock_guard<std::mutex> lock2(frame_mutex_);

        std::cout << "\n[RasterizerMemoryArena] ⚠️  EMERGENCY CLEANUP ⚠️" << std::endl;

        // Clear all inactive frames
        auto it = frame_contexts_.begin();
        while (it != frame_contexts_.end()) {
            if (!it->second.is_active) {
                it = frame_contexts_.erase(it);
            } else {
                ++it;
            }
        }

        // Reset all arena offsets and decommit unused memory
        for (auto& [device, arena] : device_arenas_) {
            if (arena) {
                arena->offset.store(0, std::memory_order_release);
                decommit_unused_memory(*arena);
                std::cout << "  Reset arena on device " << device
                          << " (committed: " << (arena->committed_size >> 20) << " MB)" << std::endl;
            }
        }

        // Try to free cached memory
        empty_cuda_cache();

        // Synchronize all devices
        for (const auto& [device, arena] : device_arenas_) {
            cudaSetDevice(device);
            cudaDeviceSynchronize();
        }

        std::cout << "[RasterizerMemoryArena] Emergency cleanup completed\n"
                  << std::endl;
    }

    RasterizerMemoryArena::Arena& RasterizerMemoryArena::get_or_create_arena(int device) {
        std::lock_guard<std::mutex> lock(arena_mutex_);

        auto& arena_ptr = device_arenas_[device];
        if (!arena_ptr) {
            arena_ptr = std::make_unique<Arena>();
            arena_ptr->device = device;
            arena_ptr->last_log_time = std::chrono::steady_clock::now();

            // Set device before allocating
            cudaSetDevice(device);

            if (is_vmm_supported(device)) {
                // Use VMM path
                auto& arena = *arena_ptr;

                // Get allocation granularity
                CUmemAllocationProp prop = {};
                prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                prop.location.id = device;

                size_t granularity = 0;
                CUresult result = cuMemGetAllocationGranularity(&granularity, &prop,
                                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM);
                if (result != CUDA_SUCCESS) {
                    // Fall back to default granularity
                    granularity = 2 << 20; // 2MB
                }
                arena.granularity = std::max(granularity, config_.granularity);

                // Reserve virtual address space (this is FREE!)
                arena.virtual_size = config_.virtual_size;
                result = cuMemAddressReserve(&arena.d_ptr, arena.virtual_size, 0, 0, 0);
                if (result != CUDA_SUCCESS) {
                    // Fall back to traditional allocation
                    std::cout << "[RasterizerMemoryArena] VMM reservation failed, using traditional allocation\n";
                    arena.d_ptr = 0;
                    arena.virtual_size = 0;
                    // Continue with traditional allocation below
                } else {
                    std::cout << "\n========================================\n"
                              << "[RasterizerMemoryArena] VMM INITIALIZATION\n"
                              << "  Device: " << device << "\n"
                              << "  Virtual space reserved: " << (arena.virtual_size >> 30)
                              << " GB (costs NO memory!)\n";

                    // Commit initial physical memory
                    if (!commit_more_memory(arena, config_.initial_commit)) {
                        // If initial commit fails, fall back to traditional
                        cuMemAddressFree(arena.d_ptr, arena.virtual_size);
                        arena.d_ptr = 0;
                        arena.virtual_size = 0;
                        std::cout << "  Initial commit failed, falling back to traditional allocation\n";
                    } else {
                        arena.generation = generation_counter_.fetch_add(1, std::memory_order_relaxed);
                        arena.offset.store(0, std::memory_order_release);
                        std::cout << "========================================\n"
                                  << std::endl;
                        return *arena_ptr;
                    }
                }
            }

            // Fallback to traditional allocation (either VMM not supported or failed)
            auto& arena = *arena_ptr;

            // Check available memory
            size_t free_memory, total_memory;
            cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to query GPU memory: " +
                                         std::string(cudaGetErrorString(err)));
            }

            // Start with a reasonable initial size
            size_t initial_size = std::min({config_.initial_commit,
                                            free_memory / 2,
                                            size_t(256) << 20});

            if (initial_size < (64 << 20)) {
                throw std::runtime_error("Insufficient GPU memory for arena initialization (need at least 64MB)");
            }

            // Try to allocate with fallback to smaller sizes
            bool allocated = false;
            while (initial_size >= (64 << 20) && !allocated) {
                err = cudaMalloc(&arena.fallback_buffer, initial_size);
                if (err == cudaSuccess) {
                    arena.capacity = initial_size;
                    arena.committed_size = initial_size;
                    arena.generation = generation_counter_.fetch_add(1, std::memory_order_relaxed);
                    arena.offset.store(0, std::memory_order_release);
                    allocated = true;

                    std::cout << "\n========================================\n"
                              << "[RasterizerMemoryArena] TRADITIONAL ALLOCATION (No VMM)\n"
                              << "  Device: " << device << "\n"
                              << "  Size: " << (initial_size >> 20) << " MB\n"
                              << "  GPU free before: " << (free_memory >> 20) << " MB\n";

                    cudaMemGetInfo(&free_memory, &total_memory);
                    std::cout << "  GPU free after: " << (free_memory >> 20) << " MB\n"
                              << "========================================\n"
                              << std::endl;
                } else {
                    initial_size /= 2;
                    std::cout << "[RasterizerMemoryArena] Allocation failed, trying "
                              << (initial_size >> 20) << " MB" << std::endl;
                }
            }

            if (!allocated) {
                throw std::runtime_error("Failed to allocate arena after multiple attempts");
            }
        }

        return *arena_ptr;
    }

    bool RasterizerMemoryArena::commit_more_memory(Arena& arena, size_t required_size) {
        // Only for VMM-enabled arenas
        if (arena.d_ptr == 0) {
            return false;
        }

        // Round up to granularity
        size_t commit_size = ((required_size + arena.granularity - 1) /
                              arena.granularity) *
                             arena.granularity;

        // Ensure we don't exceed limits
        if (arena.committed_size + commit_size > config_.max_physical) {
            // Try to commit as much as possible
            commit_size = config_.max_physical - arena.committed_size;
            if (commit_size < arena.granularity) {
                return false; // Can't commit anything
            }
            // Round down to granularity
            commit_size = (commit_size / arena.granularity) * arena.granularity;
        }

        // Check if we'd exceed virtual space
        if (arena.committed_size + commit_size > arena.virtual_size) {
            return false;
        }

        // Check available GPU memory with larger buffer
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);

        // Need substantial buffer for VMM and other operations
        size_t buffer_needed = std::min(size_t(1) << 30, total_memory / 10); // 1GB or 10% of total

        if (free_memory < commit_size + buffer_needed) {
            // Try cleanup
            empty_cuda_cache();
            cudaMemGetInfo(&free_memory, &total_memory);

            if (free_memory < commit_size + buffer_needed) {
                // Try smaller allocation
                commit_size = free_memory - buffer_needed;
                commit_size = (commit_size / arena.granularity) * arena.granularity;
                if (commit_size < arena.granularity) {
                    return false;
                }
            }
        }

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = arena.device;

        CUmemGenericAllocationHandle handle;
        CUresult result = cuMemCreate(&handle, commit_size, &prop, 0);
        if (result != CUDA_SUCCESS) {
            return false;
        }

        // Map with proper alignment
        size_t map_offset = arena.committed_size;
        map_offset = (map_offset + arena.granularity - 1) & ~(arena.granularity - 1);

        result = cuMemMap(arena.d_ptr + map_offset, commit_size, 0, handle, 0);
        if (result != CUDA_SUCCESS) {
            cuMemRelease(handle);
            return false;
        }

        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = arena.device;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        result = cuMemSetAccess(arena.d_ptr + map_offset, commit_size, &access_desc, 1);
        if (result != CUDA_SUCCESS) {
            cuMemUnmap(arena.d_ptr + map_offset, commit_size);
            cuMemRelease(handle);
            return false;
        }

        // Track the chunk
        {
            std::lock_guard<std::mutex> lock(arena.chunks_mutex);
            arena.chunks.push_back({handle, map_offset, commit_size, true});
        }

        arena.committed_size = map_offset + commit_size;
        arena.capacity = arena.committed_size;
        arena.realloc_count.fetch_add(1, std::memory_order_relaxed);

        std::cout << "[RasterizerMemoryArena] Committed " << (commit_size >> 20)
                  << " MB (total: " << (arena.committed_size >> 20) << " MB)\n";

        return true;
    }

    void RasterizerMemoryArena::decommit_unused_memory(Arena& arena) {
        // Called with arena_mutex_ held
        // For now, we don't decommit to avoid fragmentation
        return;
    }

    char* RasterizerMemoryArena::allocate_internal(Arena& arena, size_t size, uint64_t frame_id) {
        size_t aligned_size = align_size(size);

        // Sanity check
        if (aligned_size > config_.max_physical) {
            throw std::runtime_error("Single allocation request " + std::to_string(aligned_size >> 20) +
                                     " MB exceeds max physical size " + std::to_string(config_.max_physical >> 30) + " GB");
        }

        // Retry loop - keep trying until we succeed or hit max retries
        const int MAX_RETRIES = 5;
        for (int retry = 0; retry < MAX_RETRIES; ++retry) {
            // Try to allocate
            size_t offset = arena.offset.fetch_add(aligned_size, std::memory_order_acq_rel);

            if (offset + aligned_size <= arena.committed_size) {
                // Success!
                char* ptr = nullptr;
                if (arena.d_ptr != 0) {
                    ptr = reinterpret_cast<char*>(arena.d_ptr) + offset;
                } else {
                    ptr = static_cast<char*>(arena.fallback_buffer) + offset;
                }

                // Update peak usage
                size_t current_usage = offset + aligned_size;
                size_t peak = arena.peak_usage.load(std::memory_order_relaxed);
                while (current_usage > peak) {
                    if (arena.peak_usage.compare_exchange_weak(peak, current_usage)) {
                        break;
                    }
                }

                size_t period_peak = arena.peak_usage_period.load(std::memory_order_relaxed);
                while (current_usage > period_peak) {
                    if (arena.peak_usage_period.compare_exchange_weak(period_peak, current_usage)) {
                        break;
                    }
                }

                BufferHandle handle;
                handle.ptr = ptr;
                handle.size = aligned_size;
                handle.generation = arena.generation;
                handle.device = arena.device;
                record_allocation(frame_id, handle);

                return ptr;
            }

            // Allocation failed - revert the offset
            arena.offset.fetch_sub(aligned_size, std::memory_order_acq_rel);

            // Try to grow the arena
            std::lock_guard<std::mutex> lock(arena_mutex_);

            // Re-check current state after getting lock
            size_t current_offset = arena.offset.load(std::memory_order_acquire);
            size_t total_needed = current_offset + aligned_size;

            // Check if someone else already grew it
            if (total_needed <= arena.committed_size) {
                continue; // Retry allocation
            }

            // We need to grow - calculate how much
            // Be VERY generous to avoid repeated growth
            size_t growth_needed = total_needed - arena.committed_size;

            // Always grow by a significant amount to handle concurrent allocations
            size_t growth_amount = std::max({
                growth_needed * 2,    // Double what's needed (for concurrent allocs)
                arena.committed_size, // Double current size
                size_t(1) << 30       // At least 1GB
            });

            // Cap at max physical
            size_t new_committed = std::min(arena.committed_size + growth_amount, config_.max_physical);
            growth_amount = new_committed - arena.committed_size;

            if (growth_amount == 0) {
                // Can't grow anymore - we're at max
                throw std::runtime_error("Arena at maximum size (" +
                                         std::to_string(config_.max_physical >> 30) + " GB), cannot allocate " +
                                         std::to_string(aligned_size >> 20) + " MB");
            }

            // Try to grow
            bool success = false;
            if (arena.d_ptr != 0) {
                // VMM path
                success = commit_more_memory(arena, growth_amount);
            } else {
                // Traditional path
                success = grow_arena(arena, new_committed);
            }

            if (!success) {
                // Check if it's a temporary failure or permanent
                if (retry < MAX_RETRIES - 1) {
                    // Try emergency cleanup and retry
                    std::cout << "[RasterizerMemoryArena] Growth failed, attempting cleanup and retry "
                              << (retry + 1) << "/" << MAX_RETRIES << "\n";

                    empty_cuda_cache();
                    cudaDeviceSynchronize();

                    // Small delay to let other operations complete
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                } else {
                    throw std::runtime_error("Failed to grow arena after " + std::to_string(MAX_RETRIES) +
                                             " attempts for allocation of " + std::to_string(size >> 20) +
                                             " MB (usage: " + std::to_string(current_offset >> 20) +
                                             " MB, committed: " + std::to_string(arena.committed_size >> 20) + " MB)");
                }
            }

            // Growth succeeded, retry allocation
        }

        // Should never reach here
        throw std::runtime_error("Allocation failed after maximum retries");
    }

    bool RasterizerMemoryArena::grow_arena(Arena& arena, size_t required_size) {
        // Called with arena_mutex_ held
        // This is the fallback for non-VMM systems

        // Get current frame for logging
        uint64_t current_frame = frame_counter_.load(std::memory_order_relaxed);

        size_t old_capacity = arena.capacity;
        size_t new_capacity = std::max(
            required_size * 2, // Double the required size for headroom
            static_cast<size_t>(arena.capacity * 1.5f));

        // Round up to 128MB boundary
        new_capacity = ((new_capacity + (128 << 20) - 1) / (128 << 20)) * (128 << 20);
        new_capacity = std::min(new_capacity, config_.max_physical);

        if (new_capacity <= arena.capacity) {
            std::cerr << "\n[RasterizerMemoryArena] ❌ CANNOT GROW - MAX SIZE REACHED\n"
                      << "  Current capacity: " << (arena.capacity >> 20) << " MB\n"
                      << "  Required: " << (required_size >> 20) << " MB\n"
                      << "  Max allowed: " << (config_.max_physical >> 30) << " GB\n"
                      << std::endl;
            return false;
        }

        // Check available memory
        size_t free_memory_before, total_memory;
        cudaError_t err = cudaMemGetInfo(&free_memory_before, &total_memory);
        if (err != cudaSuccess) {
            return false;
        }

        size_t additional_needed = new_capacity - arena.capacity;

        // ALWAYS LOG GROWTH ATTEMPT
        std::cout << "\n========================================\n"
                  << "[RasterizerMemoryArena] 📈 GROWING ARENA (Traditional)\n"
                  << "  Frame: " << current_frame << "\n"
                  << "  Device: " << arena.device << "\n"
                  << "  Current capacity: " << (old_capacity >> 20) << " MB\n"
                  << "  Required size: " << (required_size >> 20) << " MB\n"
                  << "  New capacity: " << (new_capacity >> 20) << " MB\n"
                  << "  Additional needed: " << (additional_needed >> 20) << " MB\n"
                  << "  GPU free: " << (free_memory_before >> 20) << " MB\n"
                  << "  Reallocation #" << (arena.realloc_count.load() + 1) << "\n";

        if (free_memory_before < additional_needed + (200 << 20)) { // Keep 200MB free
            std::cout << "  ⚠️  Low memory - attempting cleanup...\n";
            // Try to free cached memory
            empty_cuda_cache();
            cudaMemGetInfo(&free_memory_before, &total_memory);
            std::cout << "  GPU free after cleanup: " << (free_memory_before >> 20) << " MB\n";

            if (free_memory_before < additional_needed + (200 << 20)) {
                std::cout << "  ❌ INSUFFICIENT MEMORY FOR GROWTH\n"
                          << "========================================\n"
                          << std::endl;
                return false;
            }
        }

        // Allocate new buffer
        void* new_buffer = nullptr;
        err = cudaMalloc(&new_buffer, new_capacity);
        if (err != cudaSuccess) {
            std::cout << "  ❌ GROWTH FAILED: " << cudaGetErrorString(err) << "\n"
                      << "========================================\n"
                      << std::endl;
            return false;
        }

        // Copy existing data
        size_t copy_size = arena.offset.load(std::memory_order_acquire);
        if (copy_size > 0 && arena.fallback_buffer) {
            err = cudaMemcpy(new_buffer, arena.fallback_buffer, copy_size, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                cudaFree(new_buffer);
                std::cout << "  ❌ COPY FAILED: " << cudaGetErrorString(err) << "\n"
                          << "========================================\n"
                          << std::endl;
                return false;
            }
        }

        // Free old buffer and replace
        if (arena.fallback_buffer) {
            cudaFree(arena.fallback_buffer);
        }
        arena.fallback_buffer = new_buffer;
        arena.capacity = new_capacity;
        arena.committed_size = new_capacity;
        arena.generation = generation_counter_.fetch_add(1, std::memory_order_relaxed);
        arena.realloc_count.fetch_add(1, std::memory_order_relaxed);

        // Get memory after allocation
        size_t free_memory_after;
        cudaMemGetInfo(&free_memory_after, &total_memory);

        std::cout << "  ✅ GROWTH SUCCESSFUL\n"
                  << "  GPU free after growth: " << (free_memory_after >> 20) << " MB\n"
                  << "  Memory used for growth: " << ((free_memory_before - free_memory_after) >> 20) << " MB\n"
                  << "========================================\n"
                  << std::endl;

        return true;
    }

    size_t RasterizerMemoryArena::align_size(size_t size) const {
        return (size + config_.alignment - 1) & ~(config_.alignment - 1);
    }

    void RasterizerMemoryArena::record_allocation(uint64_t frame_id, const BufferHandle& handle) {
        std::lock_guard<std::mutex> lock(frame_mutex_);

        auto it = frame_contexts_.find(frame_id);
        if (it != frame_contexts_.end()) {
            it->second.buffers.push_back(handle);
            it->second.total_allocated += handle.size;
        }
    }

    RasterizerMemoryArena::Statistics RasterizerMemoryArena::get_statistics() const {
        Statistics stats;

        std::lock_guard<std::mutex> lock(arena_mutex_);

        for (const auto& [device, arena_ptr] : device_arenas_) {
            if (arena_ptr) {
                stats.current_usage += arena_ptr->offset.load(std::memory_order_relaxed);
                stats.peak_usage = std::max(stats.peak_usage,
                                            arena_ptr->peak_usage.load(std::memory_order_relaxed));
                stats.capacity += arena_ptr->committed_size;
                stats.reallocation_count += arena_ptr->realloc_count.load(std::memory_order_relaxed);
            }
        }

        stats.frame_count = total_frames_processed_.load(std::memory_order_relaxed);
        stats.utilization_ratio = stats.capacity > 0 ? static_cast<float>(stats.current_usage) / static_cast<float>(stats.capacity) : 0.0f;

        return stats;
    }

    RasterizerMemoryArena::MemoryInfo RasterizerMemoryArena::get_memory_info() const {
        MemoryInfo info;

        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err == cudaSuccess) {
            std::lock_guard<std::mutex> lock(arena_mutex_);
            auto it = device_arenas_.find(device);
            if (it != device_arenas_.end() && it->second) {
                info.arena_capacity = it->second->committed_size;
                info.current_usage = it->second->offset.load(std::memory_order_relaxed);
                info.peak_usage = it->second->peak_usage.load(std::memory_order_relaxed);
                info.num_reallocations = it->second->realloc_count.load(std::memory_order_relaxed);
                info.utilization_percent = info.arena_capacity > 0 ? (100.0f * info.peak_usage / info.arena_capacity) : 0.0f;
            }
        }

        cudaMemGetInfo(&info.gpu_free, &info.gpu_total);
        return info;
    }

    void RasterizerMemoryArena::dump_statistics() const {
        auto stats = get_statistics();

        std::stringstream ss;
        ss << "\n========================================\n"
           << "[RasterizerMemoryArena] FINAL STATISTICS\n"
           << "  Total physical committed: " << (stats.capacity >> 20) << " MB\n"
           << "  Peak usage: " << (stats.peak_usage >> 20) << " MB\n"
           << "  Frames processed: " << stats.frame_count << "\n"
           << "  Total reallocations: " << stats.reallocation_count << "\n"
           << "  Peak utilization: " << std::fixed << std::setprecision(1)
           << (stats.peak_usage * 100.0f / std::max(size_t(1), stats.capacity)) << "%\n";

        auto runtime = std::chrono::steady_clock::now() - creation_time_;
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(runtime).count();
        ss << "  Runtime: " << seconds << " seconds\n";

        if (stats.frame_count > 0) {
            ss << "  Average time per frame: "
               << (seconds * 1000.0 / stats.frame_count) << " ms\n";
        }

        ss << "========================================\n";

        std::cout << ss.str() << std::flush;
    }

    bool RasterizerMemoryArena::is_under_memory_pressure() const {
        return get_memory_pressure() > 0.8f;
    }

    float RasterizerMemoryArena::get_memory_pressure() const {
        size_t free_memory, total_memory;
        cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
        if (err != cudaSuccess) {
            return 1.0f;
        }
        return 1.0f - (static_cast<float>(free_memory) / static_cast<float>(total_memory));
    }

    // Global singleton implementation
    GlobalArenaManager& GlobalArenaManager::instance() {
        static GlobalArenaManager instance;
        return instance;
    }

    RasterizerMemoryArena& GlobalArenaManager::get_arena() {
        std::lock_guard<std::mutex> lock(init_mutex_);

        if (!arena_) {
            // Create with VMM-optimized settings
            RasterizerMemoryArena::Config config;
            config.virtual_size = 32ULL << 30; // 32GB virtual (costs nothing!)
            config.initial_commit = 512 << 20; // 512MB initial physical (was 256MB)
            config.max_physical = 8ULL << 30;  // 8GB max physical
            config.granularity = 2 << 20;      // 2MB chunks
            config.alignment = 256;
            config.enable_profiling = false;
            config.log_interval = 1000;

            arena_ = std::make_unique<RasterizerMemoryArena>(config);
        }
        return *arena_;
    }

    void GlobalArenaManager::reset() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        arena_.reset();
    }

} // namespace lfs::rendering