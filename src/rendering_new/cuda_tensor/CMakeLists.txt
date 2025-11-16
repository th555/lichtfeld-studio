# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# CUDA rasterization backend for rendering module (Tensor-based, libtorch-free)
# Includes all necessary CUDA implementation files
set(CUDA_RASTERIZER_TENSOR_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/forward.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/rasterization_api_tensor.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/rasterizer_memory_arena.cu
)

# Create internal CUDA tensor rasterization library (libtorch-free)
add_library(rendering_cuda_tensor_backend STATIC ${CUDA_RASTERIZER_TENSOR_SOURCES})

set_target_properties(rendering_cuda_tensor_backend PROPERTIES
        CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(rendering_cuda_tensor_backend
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/include
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(rendering_cuda_tensor_backend
        PUBLIC
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        CUDA::cuda_driver
        spdlog::spdlog
        # gs_core  # Temporarily disabled - requires LibTorch
)

target_compile_options(rendering_cuda_tensor_backend PRIVATE
        # CUDA
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221>

        # CUDA on Windows: Pass /utf-8 to MSVC when invoked by nvcc
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:
            -Xcompiler=/utf-8
            -Xcompiler=/D_CRT_SECURE_NO_WARNINGS
            --diag-suppress=27  # Suppress "character value out of range" warnings from fmt
        >

        # C++ MSVC
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od /Z7>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /DNDEBUG>

        # C++ non-MSVC
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

# Windows: Disable fmt Unicode check for CUDA files (nvcc doesn't properly propagate /utf-8)
if(MSVC)
    target_compile_definitions(rendering_cuda_tensor_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(rendering_cuda_tensor_backend PRIVATE _DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(rendering_cuda_tensor_backend PRIVATE RELEASE_BUILD)
endif()
