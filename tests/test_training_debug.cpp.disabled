// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

// Need complete types for datasets to call methods
#include "../src/training/dataset.hpp"
#include "../src/training_new/dataset.hpp"

// Need complete types for strategies to access methods
#include "../src/training/strategies/mcmc.hpp"
#include "../src/training_new/strategies/mcmc.hpp"

#include "training_debug/training_debug.hpp"

/**
 * @brief Test fixture for training_debug module
 */
class TrainingDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set log level to info for detailed output
        spdlog::set_level(spdlog::level::info);
    }

    void TearDown() override {
        // Clean up if needed
    }
};

/**
 * @brief Test loading dataset with legacy loader
 */
TEST_F(TrainingDebugTest, LoadDatasetLegacy) {
    auto result = gs::training_debug::load_dataset_legacy();
    ASSERT_TRUE(result.has_value()) << "Legacy dataset loading failed: " << result.error();
}

/**
 * @brief Test loading dataset with new loader
 */
TEST_F(TrainingDebugTest, LoadDatasetNew) {
    auto result = gs::training_debug::load_dataset_new();
    ASSERT_TRUE(result.has_value()) << "New dataset loading failed: " << result.error();
}

/**
 * @brief Test initializing legacy training pipeline
 */
TEST_F(TrainingDebugTest, InitializeLegacy) {
    auto result = gs::training_debug::initialize_legacy();
    ASSERT_TRUE(result.has_value()) << "Legacy initialization failed: " << result.error();

    // Verify initialization data
    EXPECT_GT(result->num_gaussians, 0) << "Legacy should have initialized Gaussians";
    EXPECT_NE(result->dataset, nullptr) << "Legacy dataset should not be null";
    EXPECT_GT(result->cam_id_to_cam.size(), 0) << "Legacy camera cache should not be empty";

    spdlog::info("Legacy initialization verified:");
    spdlog::info("  - Gaussians: {}", result->num_gaussians);
    spdlog::info("  - Cameras: {}", result->dataset->size().value());
    spdlog::info("  - Camera cache size: {}", result->cam_id_to_cam.size());
}

/**
 * @brief Test initializing new training pipeline
 */
TEST_F(TrainingDebugTest, InitializeNew) {
    auto result = gs::training_debug::initialize_new();
    ASSERT_TRUE(result.has_value()) << "New initialization failed: " << result.error();

    // Verify initialization data
    EXPECT_GT(result->num_gaussians, 0) << "New should have initialized Gaussians";
    EXPECT_NE(result->dataset, nullptr) << "New dataset should not be null";
    EXPECT_GT(result->cam_id_to_cam.size(), 0) << "New camera cache should not be empty";

    spdlog::info("New initialization verified:");
    spdlog::info("  - Gaussians: {}", result->num_gaussians);
    spdlog::info("  - Cameras: {}", result->dataset->size());
    spdlog::info("  - Camera cache size: {}", result->cam_id_to_cam.size());
}

/**
 * @brief Test initializing both pipelines and comparing
 */
TEST_F(TrainingDebugTest, InitializeBoth) {
    auto result = gs::training_debug::initialize_both();
    ASSERT_TRUE(result.has_value()) << "Combined initialization failed: " << result.error();

    auto& [legacy, new_impl] = *result;

    // Verify both initialized successfully
    EXPECT_GT(legacy.num_gaussians, 0) << "Legacy should have Gaussians";
    EXPECT_GT(new_impl.num_gaussians, 0) << "New should have Gaussians";

    // Compare initialization results
    EXPECT_EQ(legacy.num_gaussians, new_impl.num_gaussians)
        << "Legacy and new should have same number of Gaussians";

    EXPECT_EQ(legacy.dataset->size().value(), new_impl.dataset->size())
        << "Legacy and new should have same number of cameras";

    EXPECT_EQ(legacy.cam_id_to_cam.size(), new_impl.cam_id_to_cam.size())
        << "Legacy and new should have same camera cache size";

    // Log comparison results
    spdlog::info("=== Comparison Results ===");
    spdlog::info("Gaussians - Legacy: {} | New: {} | Match: {}",
                 legacy.num_gaussians,
                 new_impl.num_gaussians,
                 legacy.num_gaussians == new_impl.num_gaussians ? "✓" : "✗");

    spdlog::info("Cameras - Legacy: {} | New: {} | Match: {}",
                 legacy.dataset->size().value(),
                 new_impl.dataset->size(),
                 legacy.dataset->size().value() == new_impl.dataset->size() ? "✓" : "✗");

    spdlog::info("Camera cache - Legacy: {} | New: {} | Match: {}",
                 legacy.cam_id_to_cam.size(),
                 new_impl.cam_id_to_cam.size(),
                 legacy.cam_id_to_cam.size() == new_impl.cam_id_to_cam.size() ? "✓" : "✗");
}

/**
 * @brief Test that both pipelines initialize with same parameters
 */
TEST_F(TrainingDebugTest, DISABLED_CompareParameters) {
    auto result = gs::training_debug::initialize_both();
    ASSERT_TRUE(result.has_value()) << "Combined initialization failed: " << result.error();

    auto& [legacy, new_impl] = *result;

    // Compare optimization parameters
    EXPECT_EQ(legacy.params.optimization.strategy, new_impl.params.optimization.strategy)
        << "Both should use same strategy";

    EXPECT_EQ(legacy.params.optimization.max_cap, new_impl.params.optimization.max_cap)
        << "Both should use same max_cap";

    EXPECT_EQ(legacy.params.dataset.data_path, new_impl.params.dataset.data_path)
        << "Both should use same data path";
}

/**
 * @brief Test rendering comparison - renders same camera from both pipelines
 */
TEST_F(TrainingDebugTest, RenderComparison) {
    spdlog::info("=== Starting RenderComparison test ===");

    // Initialize both pipelines
    auto init_result = gs::training_debug::initialize_both();
    ASSERT_TRUE(init_result.has_value()) << "Initialization failed: " << init_result.error();

    auto& [legacy, new_impl] = *init_result;

    // Verify we have cameras to render
    ASSERT_GT(legacy.dataset->size().value(), 0) << "Legacy dataset has no cameras";
    ASSERT_GT(new_impl.dataset->size(), 0) << "New dataset has no cameras";

    // Use the first camera (index 0) for rendering
    size_t camera_index = 0;

    spdlog::info("Rendering camera {} from both pipelines...", camera_index);

    // Render and save comparison
    auto render_result = gs::training_debug::render_and_save_comparison(
        legacy,
        new_impl,
        camera_index,
        "/tmp/render_comparison_test.png"
    );

    ASSERT_TRUE(render_result.has_value()) << "Rendering comparison failed: " << render_result.error();

    // Check that output files were created
    EXPECT_TRUE(std::filesystem::exists("/tmp/render_comparison_test_legacy.png"))
        << "Legacy comparison image not saved";
    EXPECT_TRUE(std::filesystem::exists("/tmp/render_comparison_test_new.png"))
        << "New comparison image not saved";

    spdlog::info("✓ Rendering comparison test passed");
    spdlog::info("  - Legacy comparison: /tmp/render_comparison_test_legacy.png");
    spdlog::info("  - New comparison: /tmp/render_comparison_test_new.png");
}

/**
 * @brief Test rendering multiple cameras
 */
TEST_F(TrainingDebugTest, RenderMultipleCameras) {
    spdlog::info("=== Starting RenderMultipleCameras test ===");

    // Initialize both pipelines
    auto init_result = gs::training_debug::initialize_both();
    ASSERT_TRUE(init_result.has_value()) << "Initialization failed: " << init_result.error();

    auto& [legacy, new_impl] = *init_result;

    // Render first 3 cameras (or all if fewer than 3)
    size_t num_cameras = std::min(size_t(3), legacy.dataset->size().value());

    spdlog::info("Rendering {} cameras from both pipelines...", num_cameras);

    for (size_t i = 0; i < num_cameras; ++i) {
        std::string output_path = std::format("/tmp/render_comparison_cam{}.png", i);

        auto render_result = gs::training_debug::render_and_save_comparison(
            legacy,
            new_impl,
            i,
            output_path
        );

        ASSERT_TRUE(render_result.has_value())
            << "Rendering camera " << i << " failed: " << render_result.error();

        spdlog::info("  ✓ Rendered camera {}", i);
    }

    spdlog::info("✓ Multiple camera rendering test passed");
}

/**
 * @brief Test training loop comparison for both pipelines
 */
TEST_F(TrainingDebugTest, TrainingLoopComparison) {
    spdlog::info("=== Starting TrainingLoopComparison test ===");

    // Initialize both pipelines
    auto init_result = gs::training_debug::initialize_both();
    ASSERT_TRUE(init_result.has_value()) << "Initialization failed: " << init_result.error();

    auto& [legacy, new_impl] = *init_result;

    // Verify we have cameras for training
    ASSERT_GT(legacy.dataset->size().value(), 0) << "Legacy dataset has no cameras";
    ASSERT_GT(new_impl.dataset->size(), 0) << "New dataset has no cameras";

    // Run training loop comparison with 3 iterations on camera 0
    size_t camera_index = 0;
    int max_iterations = 3;

    spdlog::info("Running training loop comparison: {} iterations on camera {}",
                 max_iterations, camera_index);

    auto train_result = gs::training_debug::run_training_loop_comparison(
        legacy, new_impl, camera_index, max_iterations);

    ASSERT_TRUE(train_result.has_value())
        << "Training loop comparison failed: " << train_result.error();

    spdlog::info("✓ Training loop comparison test passed");
}

/**
 * @brief Test training loop comparison with longer sequence
 */
TEST_F(TrainingDebugTest, DISABLED_TrainingLoopLonger) {
    spdlog::info("=== Starting TrainingLoopLonger test ===");

    // Initialize both pipelines
    auto init_result = gs::training_debug::initialize_both();
    ASSERT_TRUE(init_result.has_value()) << "Initialization failed: " << init_result.error();

    auto& [legacy, new_impl] = *init_result;

    // Run training loop comparison with 10 iterations
    size_t camera_index = 0;
    int max_iterations = 10;

    spdlog::info("Running extended training loop comparison: {} iterations", max_iterations);

    auto train_result = gs::training_debug::run_training_loop_comparison(
        legacy, new_impl, camera_index, max_iterations);

    ASSERT_TRUE(train_result.has_value())
        << "Training loop comparison failed: " << train_result.error();

    spdlog::info("✓ Extended training loop comparison test passed");
}

/**
 * @brief Test SH0 initialization values between legacy and new implementations
 */
TEST_F(TrainingDebugTest, SH0InitializationComparison) {
    spdlog::info("=== Starting SH0 Initialization Comparison ===");

    auto result = gs::training_debug::initialize_both();
    ASSERT_TRUE(result.has_value()) << "Initialization failed: " << result.error();

    auto& [legacy, new_impl] = *result;

    // Get SH0 values
    auto legacy_sh0_raw = legacy.strategy->get_model().sh0().cpu();
    auto new_sh0 = new_impl.strategy->get_model().sh0();

    // Legacy sh0 is [N, 1, 3], need to reshape to [N, 3] for comparison
    auto legacy_sh0 = legacy_sh0_raw.reshape({legacy_sh0_raw.size(0), 3});

    auto new_sh0_flat = new_sh0.contiguous().flatten(1, 2);
    auto new_sh0_cpu = new_sh0_flat.to(lfs::core::Device::CPU);

    // Convert to torch - manual memcpy
    size_t N = new_sh0_cpu.shape()[0];
    size_t C = new_sh0_cpu.shape()[1];
    torch::Tensor new_sh0_torch = torch::empty({static_cast<long>(N), static_cast<long>(C)}, torch::kFloat32);
    const float* src_ptr = new_sh0_cpu.ptr<float>();
    float* dst_ptr = new_sh0_torch.data_ptr<float>();
    std::memcpy(dst_ptr, src_ptr, N * C * sizeof(float));

    auto diff_tensor = (legacy_sh0 - new_sh0_torch).abs();
    float max_diff = diff_tensor.max().item().toFloat();
    float mean_diff = diff_tensor.mean().item().toFloat();

    spdlog::info("=== SH0 Initial Values Comparison ===");
    spdlog::info("Max diff: {:.6f}, Mean diff: {:.6f}", max_diff, mean_diff);
    spdlog::info("Legacy SH0 shape: [{}, {}]", legacy_sh0.size(0), legacy_sh0.size(1));
    spdlog::info("New SH0 shape: [{}, {}]", new_sh0_torch.size(0), new_sh0_torch.size(1));

    // Print first few values
    spdlog::info("First 3 Legacy SH0 values:");
    for (int i = 0; i < 3; ++i) {
        auto row = legacy_sh0[i];
        spdlog::info("  [{}]: [{:.6f}, {:.6f}, {:.6f}]",
                     i,
                     row[0].item().toFloat(),
                     row[1].item().toFloat(),
                     row[2].item().toFloat());
    }

    spdlog::info("First 3 New SH0 values:");
    for (int i = 0; i < 3; ++i) {
        auto row = new_sh0_torch[i];
        spdlog::info("  [{}]: [{:.6f}, {:.6f}, {:.6f}]",
                     i,
                     row[0].item().toFloat(),
                     row[1].item().toFloat(),
                     row[2].item().toFloat());
    }

    // Check statistics
    spdlog::info("Legacy SH0 - Min: {:.6f}, Max: {:.6f}, Mean: {:.6f}",
                 legacy_sh0.min().item().toFloat(),
                 legacy_sh0.max().item().toFloat(),
                 legacy_sh0.mean().item().toFloat());
    spdlog::info("New SH0 - Min: {:.6f}, Max: {:.6f}, Mean: {:.6f}",
                 new_sh0_torch.min().item().toFloat(),
                 new_sh0_torch.max().item().toFloat(),
                 new_sh0_torch.mean().item().toFloat());
}
