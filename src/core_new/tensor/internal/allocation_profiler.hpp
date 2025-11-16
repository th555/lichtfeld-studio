/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifdef __linux__
#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>
#endif

namespace lfs::core {

// Enable/disable profiling at compile time (controlled by CMake)
#ifndef ENABLE_ALLOCATION_PROFILING
#define ENABLE_ALLOCATION_PROFILING 0  // Default to disabled if not set by CMake
#endif

struct AllocationSite {
    size_t total_bytes = 0;
    size_t count = 0;
    size_t peak_bytes = 0;

    void record(size_t bytes) {
        total_bytes += bytes;
        count++;
        if (bytes > peak_bytes) {
            peak_bytes = bytes;
        }
    }
};

class AllocationProfiler {
public:
    static AllocationProfiler& instance() {
        static AllocationProfiler profiler;
        return profiler;
    }

    // Capture stack trace and record allocation
    void record_allocation(size_t bytes, int skip_frames = 2) {
        if constexpr (!ENABLE_ALLOCATION_PROFILING) {
            return;
        }

#ifdef __linux__
        // Capture stack trace
        constexpr int MAX_FRAMES = 20;
        void* callstack[MAX_FRAMES];
        int frames = backtrace(callstack, MAX_FRAMES);

        if (frames <= skip_frames) {
            return;
        }

        // Get symbols for the entire stack
        char** symbols = backtrace_symbols(callstack, frames);
        if (!symbols) {
            return;
        }

        // Build location string from multiple frames to get context
        std::string location;

        // Start from skip_frames and collect up to 10 frames (full call chain)
        for (int i = skip_frames; i < std::min(frames, skip_frames + 10); ++i) {
            std::string frame = symbols[i];

            // Extract function name between '(' and '+' or ')'
            size_t start = frame.find('(');
            size_t end = frame.find('+', start);
            if (end == std::string::npos) {
                end = frame.find(')', start);
            }

            if (start != std::string::npos && end != std::string::npos && end > start + 1) {
                std::string mangled = frame.substr(start + 1, end - start - 1);

                // Try to demangle
                int status = 0;
                char* demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
                if (status == 0 && demangled) {
                    // Shorten long template names
                    std::string func = demangled;
                    free(demangled);

                    // Remove template noise for readability
                    size_t template_start = func.find('<');
                    if (template_start != std::string::npos) {
                        func = func.substr(0, template_start);
                    }

                    if (!location.empty()) location += " <- ";
                    location += func;
                } else if (!mangled.empty()) {
                    if (!location.empty()) location += " <- ";
                    location += mangled;
                }
            }
        }

        free(symbols);

        // Fallback if we couldn't extract anything
        if (location.empty()) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%p", callstack[skip_frames]);
            location = buf;
        }

        // Record allocation
        std::lock_guard<std::mutex> lock(mutex_);
        sites_[location].record(bytes);
        total_allocs_++;
        total_bytes_ += bytes;
#endif
    }

    // Print top N allocation sites
    void print_top_allocators(int top_n = 20) {
        if constexpr (!ENABLE_ALLOCATION_PROFILING) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Convert map to vector for sorting
        std::vector<std::pair<std::string, AllocationSite>> sites_vec(sites_.begin(), sites_.end());

        // Sort by total bytes
        std::sort(sites_vec.begin(), sites_vec.end(),
                  [](const auto& a, const auto& b) {
                      return a.second.total_bytes > b.second.total_bytes;
                  });

        printf("\n========== TOP %d ALLOCATION SITES ==========\n", top_n);
        printf("Total allocations: %zu, Total bytes: %.2f GB\n\n",
               total_allocs_.load(), total_bytes_.load() / (1024.0 * 1024.0 * 1024.0));

        printf("%-80s | %12s | %8s | %12s\n", "Origin", "Total (MB)", "Count", "Avg (KB)");
        printf("%s\n", std::string(120, '-').c_str());

        int count = 0;
        for (const auto& [location, site] : sites_vec) {
            if (count++ >= top_n) break;

            double total_mb = site.total_bytes / (1024.0 * 1024.0);
            double avg_kb = (site.total_bytes / site.count) / 1024.0;

            // Extract the origin (rightmost 2 functions in call chain)
            std::string origin = location;
            size_t last_arrow = location.rfind(" <- ");
            if (last_arrow != std::string::npos) {
                // Find the second-to-last arrow to get 2 functions
                size_t second_last = location.rfind(" <- ", last_arrow - 1);
                if (second_last != std::string::npos) {
                    origin = location.substr(second_last + 4);  // +4 to skip " <- "
                } else {
                    origin = location.substr(last_arrow + 4);
                }
            }

            // Truncate if still too long
            if (origin.length() > 80) {
                origin = "..." + origin.substr(origin.length() - 77);
            }

            printf("%-80s | %12.2f | %8zu | %12.2f\n",
                   origin.c_str(),
                   total_mb,
                   site.count,
                   avg_kb);
        }
        printf("\n");
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        sites_.clear();
        total_allocs_ = 0;
        total_bytes_ = 0;
    }

private:
    AllocationProfiler() = default;

    std::mutex mutex_;
    std::map<std::string, AllocationSite> sites_;
    std::atomic<size_t> total_allocs_{0};
    std::atomic<size_t> total_bytes_{0};
};

} // namespace lfs::core
