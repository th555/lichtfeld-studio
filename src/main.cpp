/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// New (LibTorch-free) implementation
#include "core_new/application.hpp"
#include "core_new/argument_parser.hpp"
#include "core_new/logger.hpp"
#include "core_new/pinned_memory_allocator.hpp"

#include <iostream>
#include <print>

int main(int argc, char* argv[]) {
    // Use new (LibTorch-free) implementation
    auto params_result = lfs::core::args::parse_args_and_params(argc, argv);
    if (!params_result) {
        LOG_ERROR("Failed to parse arguments: {}", params_result.error());
        std::println(stderr, "Error: {}", params_result.error());
        return -1;
    }

    // Logger is now ready to use
    LOG_INFO("========================================");
    LOG_INFO("LichtFeld Studio");
    LOG_INFO("========================================");

    // Pre-warm pinned memory cache to avoid cudaHostAlloc overhead during training
    lfs::core::PinnedMemoryAllocator::instance().prewarm();

    auto params = std::move(*params_result);

    lfs::core::Application app;
    return app.run(std::move(params));
}
