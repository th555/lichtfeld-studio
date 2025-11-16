/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Legacy (LibTorch-based) implementation
#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/logger.hpp"

#include <iostream>
#include <print>

int main(int argc, char* argv[]) {
    // Use legacy (LibTorch-based) implementation
    auto params_result = gs::args::parse_args_and_params(argc, argv);
    if (!params_result) {
        LOG_ERROR("Failed to parse arguments: {}", params_result.error());
        std::println(stderr, "Error: {}", params_result.error());
        return -1;
    }

    LOG_INFO("========================================");
    LOG_INFO("LichtFeld Studio (LEGACY)");
    LOG_INFO("========================================");

    auto params = std::move(*params_result);

    gs::Application app;
    return app.run(std::move(params));
}
