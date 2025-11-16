/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace lfs::vis {
    // Forward declarations
    class VisualizerImpl;

    namespace gui {
        class FileBrowser;

        // Shared context passed to all UI functions
        struct UIContext {
            VisualizerImpl* viewer;
            FileBrowser* file_browser;
            std::unordered_map<std::string, bool>* window_states;
        };
    } // namespace gui
} // namespace lfs::vis