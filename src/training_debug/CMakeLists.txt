# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Training debug module - for comparing old (gs) vs new (lfs) training implementations
# This module can access both legacy training and new training_new modules

set(TRAINING_DEBUG_SOURCES
    # Debug/comparison source files
    training_debug.cpp
)

# Create training debug library
add_library(gs_training_debug STATIC ${TRAINING_DEBUG_SOURCES})

target_include_directories(gs_training_debug
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(gs_training_debug
    PUBLIC
        # Link both training modules
        gs_training              # Legacy training
        lfs_training             # New training

        # Core modules
        gs_core
        lfs_core

        # Other dependencies
        ${TORCH_LIBRARIES}
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        glm::glm

    PRIVATE
        CUDA::cudart
)

# Set C++ standard
set_target_properties(gs_training_debug PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    DEBUG_POSTFIX d
)

# Compiler options
if (MSVC)
    target_compile_options(gs_training_debug PRIVATE
        $<$<CONFIG:Debug>:/Od /Z7>
        $<$<CONFIG:Release>:/O2 /DNDEBUG>)
else()
    target_compile_options(gs_training_debug PRIVATE
        $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer -DDEBUG>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>)
endif()

message(STATUS "╔════════════════════════════════════════════════════════════════╗")
message(STATUS "║  gs_training_debug - Training Comparison & Debug Module       ║")
message(STATUS "╚════════════════════════════════════════════════════════════════╝")
message(STATUS "  • Purpose: Compare gs::training vs lfs::training implementations")
message(STATUS "  • Namespace: gs::training_debug (suggested)")
message(STATUS "  • Links: gs_training (legacy) + lfs_training (new)")
