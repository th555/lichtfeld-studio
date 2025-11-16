# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Project management module (LibTorch-free)
set(PROJECT_SOURCES
    project.cpp
)

add_library(lfs_project STATIC ${PROJECT_SOURCES})

target_include_directories(lfs_project
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(lfs_project
    PUBLIC
        nlohmann_json::nlohmann_json
    PRIVATE
        spdlog::spdlog
)

set_target_properties(lfs_project PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    DEBUG_POSTFIX d
)
