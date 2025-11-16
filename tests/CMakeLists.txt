# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Testing module - comprehensive tests for all components

# Find GTest
find_package(GTest CONFIG REQUIRED)

# Note: WebP and LibArchive are already found by parent CMakeLists.txt
# and will be available through gs_core target

# Test sources - START WITH ONLY EXISTING FILES
set(TEST_SOURCES
    test_main.cpp       # Custom main with pinned memory pre-warming
    test_geometry.cpp
    test_ssim_kernel_wrapper.cu  # CUDA wrapper for SSIM kernel comparison tests
)

# Check if new test files exist and add them
set(OPTIONAL_TEST_FILES
    test_loss_gradients.cpp
    test_regularization_comparison.cpp
    test_add_new_gs_deterministic.cpp
    test_render_modes.cpp
    test_gut_rasterizer_gradients.cpp
    test_gut_manual_vs_autograd.cpp
    test_gut_autograd_vs_manual.cpp
    test_gut_gradient_comparison.cpp
    test_activation_gradients.cpp
    test_tensor_basic.cpp
    test_tensor_ops.cpp
    test_tensor_memory.cpp
    test_tensor_reduction.cpp
    test_tensor_views.cpp
    test_tensor_math.cpp
    test_tensor_utils.cpp
    test_tensor_torch_compat.cpp
    test_tensor_advanced.cpp
    test_tensor_stress.cpp
    test_tensor_matrix.cpp
    test_tensor_random.cpp
    test_tensor_broadcast.cpp
    test_tensor_masking.cpp
    test_tensor_bitwise.cpp
    test_tensor_accessor.cpp
    test_tensor_random_advanced.cpp
    test_tensor_indexing_advanced.cpp
    test_tensor_clamping.cpp
    test_tensor_compat.cpp
    test_tensor_cumsum_convenience.cpp
    test_tensor_conversions_shape.cpp
    test_tensor_reserve_inplace_cat.cpp
    test_kmeans.cpp
    test_critical_fixes.cpp
    test_cuda_device_state.cpp
    test_tensor_ops_comparison.cpp
    test_sort_minimal.cpp
    test_tensor_debug.cpp
    test_bugs_found.cpp
    test_tensor_vs_torch.cpp
    test_tensor_move.cpp
    test_mcmc.cpp
    test_loader_comparison.cpp  # Comprehensive comparison between old and new loaders
    test_ply_loading_comparison.cpp  # PLY loading comparison between legacy and new
    test_ply_save_comparison.cpp  # PLY saving comparison between legacy and new
    test_mcmc_add_new_gs_comparison.cpp  # Compare MCMC add_new_gs between legacy and new
    test_mcmc_relocate_optimizer_state_bug.cpp  # Test optimizer state bug in relocate_gs
    test_training_debug.cpp     # Training debug module - compare legacy vs new training
    test_tensor_benchmark.cpp
    test_memory_pool_benchmark.cpp  # benchmarks CudaMemoryPool (now internal)
    test_broadcast_benchmark.cpp
    test_reduction_benchmark.cpp
    test_expression_templates.cpp
    test_fusion_benchmark.cpp
    test_gather_benchmark.cpp
    test_zip_gather_benchmark.cpp
    test_transform_iterator_benchmark.cpp
    test_cub_reduction_benchmark.cpp
    benchmark_strided_tensors.cpp
    benchmark_grad_alpha_layout.cpp
    benchmark_background_blend.cpp
    test_scalar_reduction_benchmark.cpp
    test_permute_upload_benchmark.cpp
    test_lfs_adam_optimizer.cpp
    test_lfs_scheduler.cpp
    benchmark_scheduler.cpp
    test_lfs_losses.cpp
    benchmark_losses.cpp
    test_mcmc_kernels.cpp
    test_mcmc_strategy.cpp
    test_tensor_bool.cpp
    test_dataloader_lfs.cpp
    test_bilateral_grid.cpp
    test_sparsity_optimizer.cpp
    test_sparsity_debug.cpp
    test_default_strategy.cpp
    test_default_strategy_comparison.cpp
    benchmark_default_strategy.cpp
    test_default_strategy_operations_comparison.cpp
    test_default_strategy_real_conditions.cpp
    test_dtype_operations_comprehensive.cpp
    test_tensor_bugs.cpp
    test_densification_performance.cpp
    test_densification_old_vs_new.cpp
    test_densification_profile.cpp
    # Merged test files for better organization
    test_bug_regressions.cpp        # Bug regressions (binary_search, index_select, minimal_tensor)
    test_torch_comparisons.cpp      # LibTorch comparisons (naive_split, split_kernel, fill)
    test_densification.cpp          # Densification tests (exact_pattern, lfs_only, benchmarks, 10M tests)
    test_mcmc_logit_verification.cpp  # Verify log/logit operations match between legacy and new
    test_metrics_comparison.cpp  # Compare old and new metrics implementations (PSNR, SSIM, no LPIPS in new)
    test_tensor_reserve_capacity.cpp  # Test tensor reserve() and capacity management
    test_tensor_cat_inplace.cpp  # Test in-place cat() optimization
    test_tensor_inplace_cat_debug.cpp  # Debug tests for in-place cat() crash
    test_tensor_slice_assignment_reserve.cpp  # Test reserve + cat as alternative to in-place slice assignment
    test_tensor_mcmc_memory_profile.cpp  # MCMC-style memory profiling with huge tensors
    test_mcmc_tensor_ops.cpp        # Comprehensive MCMC tensor operation tests (nonzero, multinomial, index_select)
    test_mcmc_memory_ops.cpp        # Consolidated: append_gather, in-place cat, memory leak detection for MCMC
    test_mcmc_critical_bugs.cpp     # Critical MCMC bugs: Adam add_new_params, inject_noise, cat self-ref, index_add_ int64
)

foreach(TEST_FILE ${OPTIONAL_TEST_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}")
        list(APPEND TEST_SOURCES ${TEST_FILE})
        message(STATUS "Found test file: ${TEST_FILE}")
    else()
        message(STATUS "Test file not found (skipping): ${TEST_FILE}")
    endif()
endforeach()

# Create test executable
add_executable(lichtfeld_tests ${TEST_SOURCES})

# Set CUDA architectures
set_target_properties(lichtfeld_tests PROPERTIES
    CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

# Include directories
# NOTE: Order matters! We want tests to explicitly choose old vs new headers
target_include_directories(lichtfeld_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src/training_new  # New libtorch-free (for test_lfs_*.cpp)
    ${CMAKE_SOURCE_DIR}/src               # Old torch-based (for existing tests)
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
)

# Define TEST_DATA_DIR for tests
target_compile_definitions(lichtfeld_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
)

# Link libraries - Fix linking order and add missing dependencies
target_link_libraries(lichtfeld_tests PRIVATE
    # Core libraries (in dependency order)
    gs_visualizer
    gs_rendering
    gs_training
    gs_training_kernels
    gs_project
    gs_geometry
    gs_core
    gs_loader  # After gs_core since gs_core depends on gs_loader
    gs_kernels

    # New torch-free modules (for comparison tests)
    lfs_visualizer
    lfs_rendering
    lfs_training
    lfs_project   # Before lfs_loader (lfs_loader uses lfs_project functions)
    lfs_core      # Before lfs_geometry (lfs_core depends on lfs_geometry)
    lfs_geometry  # After lfs_core - provides EuclideanTransform
    lfs_loader  # After lfs_project and lfs_core
    lfs_tensor
    lfs_project   # Second pass - lfs_loader depends on lfs_project

    # Backend libraries
    gsplat_backend
    fastgs_backend

    # External libraries (using custom test_main.cpp instead of gtest_main)
    GTest::gtest
    ${TORCH_LIBRARIES}
    ${OPENGL_LIBRARIES}
    OpenImageIO::OpenImageIO

    # CUDA libraries
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::cupti      # Required by PyTorch profiling

    # System libraries
    spdlog::spdlog
)

# Compiler options for CUDA debugging
target_compile_options(lichtfeld_tests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo -Xcudafe --device-debug>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

# Enable testing
include(GoogleTest)
gtest_discover_tests(lichtfeld_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES TIMEOUT 30
)

# Custom target to run tests with output
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
    DEPENDS lichtfeld_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests with verbose output"
)

# Note: Profiling executables removed during test cleanup

# Platform-specific settings
if(WIN32)
    # Copy Torch DLLs for testing on Windows
    file(GLOB TORCH_DLLS "${Torch_DIR}/../../../lib/*.dll")
    foreach(TORCH_DLL ${TORCH_DLLS})
        add_custom_command(
            TARGET lichtfeld_tests
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_DLL}"
            "$<TARGET_FILE_DIR:lichtfeld_tests>")
    endforeach()
endif()

message(STATUS "Tests configured with ${CMAKE_BUILD_TYPE} build type")
message(STATUS "  Test sources found: ${TEST_SOURCES}")
message(STATUS "  Build with: ninja lichtfeld_tests")
message(STATUS "  Run with: ninja run_tests or ctest --output-on-failure")
