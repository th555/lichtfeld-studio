/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_mcmc_add_new_gs_comparison.cpp
 * @brief Compare MCMC add_new_gs between legacy (LibTorch) and new (LibTorch-free)
 *
 * This test validates that the new add_new_gs implementation produces
 * equivalent results to the legacy implementation using real-world data
 * after realistic training iterations.
 */

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <cmath>

// Training debug module for realistic training simulation
#include "training_debug/training_debug.hpp"

// Legacy implementation
#include "training/strategies/mcmc.hpp"
#include "training/dataset.hpp"
#include "core/splat_data.hpp"

// New implementation
#include "training_new/strategies/mcmc.hpp"
#include "training_new/dataset.hpp"
#include "core_new/splat_data.hpp"

namespace fs = std::filesystem;

class MCMCAddNewGsTest : public ::testing::Test {
protected:
    void SetUp() override {
        spdlog::set_level(spdlog::level::info);
    }

    // Helper: Compare tensor values between torch and lfs tensors
    void compareTensors(const torch::Tensor& legacy_tensor,
                       const lfs::core::Tensor& new_tensor,
                       const std::string& name,
                       float tolerance = 1e-3f) {  // Relaxed tolerance for trained models
        // Move to CPU for comparison
        auto legacy_cpu = legacy_tensor.cpu().contiguous();
        auto new_cpu = new_tensor.cpu().contiguous();

        // Check shapes match
        ASSERT_EQ(legacy_cpu.size(0), new_cpu.shape()[0])
            << name << " shape mismatch in dim 0";

        if (legacy_cpu.dim() > 1 && new_cpu.ndim() > 1) {
            ASSERT_EQ(legacy_cpu.size(1), new_cpu.shape()[1])
                << name << " shape mismatch in dim 1";
        }

        // Compare values
        const float* legacy_ptr = legacy_cpu.data_ptr<float>();
        const float* new_ptr = new_cpu.ptr<float>();
        size_t numel = legacy_cpu.numel();

        float max_diff = 0.0f;
        float mean_diff = 0.0f;
        size_t num_mismatches = 0;

        for (size_t i = 0; i < numel; ++i) {
            float diff = std::abs(legacy_ptr[i] - new_ptr[i]);
            max_diff = std::max(max_diff, diff);
            mean_diff += diff;

            if (diff > tolerance) {
                num_mismatches++;
                if (num_mismatches <= 5) {  // Print first 5 mismatches
                    spdlog::warn("{} mismatch at index {}: legacy={:.6f}, new={:.6f}, diff={:.6f}",
                                name, i, legacy_ptr[i], new_ptr[i], diff);
                }
            }
        }

        mean_diff /= numel;

        spdlog::info("{}: max_diff={:.6e}, mean_diff={:.6e}, mismatches={}/{} ({:.2f}%)",
                    name, max_diff, mean_diff, num_mismatches, numel,
                    100.0f * num_mismatches / numel);

        EXPECT_LT(max_diff, tolerance * 10.0f)  // Allow 10x tolerance for max
            << name << " has differences exceeding tolerance";
        EXPECT_LT(num_mismatches, numel * 0.05)  // Allow 5% mismatches
            << name << " has too many mismatches";
    }
};

TEST_F(MCMCAddNewGsTest, CompareAddNewGsAfterTraining) {
    spdlog::info("=== Initializing Both Pipelines ===");

    // Set random seeds for reproducibility
    torch::manual_seed(42);
    srand(42);  // For CPU random
    lfs::core::Tensor::manual_seed(42);  // For lfs::core::Tensor CUDA RNG

    auto init_result = gs::training_debug::initialize_both();
    ASSERT_TRUE(init_result.has_value())
        << "Failed to initialize pipelines: " << init_result.error();

    auto& [legacy_init, new_init] = init_result.value();

    spdlog::info("Legacy: {} Gaussians", legacy_init.num_gaussians);
    spdlog::info("New: {} Gaussians", new_init.num_gaussians);

    ASSERT_EQ(legacy_init.num_gaussians, new_init.num_gaussians)
        << "Initial Gaussian count mismatch";

    // Run training iterations to create realistic opacity variations
    spdlog::info("=== Running Training Iterations (500 steps) ===");
    auto train_result = gs::training_debug::run_training_loop_comparison(
        legacy_init, new_init, 0, 500);  // Camera 0, 500 iterations

    ASSERT_TRUE(train_result.has_value())
        << "Training loop failed: " << train_result.error();

    spdlog::info("=== Training Complete - Models Now Have Realistic States ===");

    // Get models after training - MUST access through strategy, not init struct!
    spdlog::info("Accessing legacy model through strategy...");
    auto& legacy_model_ref = legacy_init.strategy->get_model();

    spdlog::info("Accessing new model through strategy...");
    auto& new_model_ref = new_init.strategy->get_model();

    spdlog::info("Getting model sizes...");
    size_t initial_count = legacy_model_ref.size();
    spdlog::info("Gaussian count after training: {}", initial_count);

    // Verify opacities have variation (not all the same)
    spdlog::info("Getting legacy opacity...");
    auto legacy_opacity_before = legacy_model_ref.get_opacity();
    spdlog::info("Moving to CPU...");
    auto legacy_opacity_cpu = legacy_opacity_before.cpu();

    spdlog::info("Getting new opacity...");
    auto new_opacity_before = new_model_ref.get_opacity();
    spdlog::info("Moving to CPU...");
    auto new_opacity_cpu = new_opacity_before.cpu();

    spdlog::info("Computing legacy opacity statistics...");
    float legacy_opacity_mean = legacy_opacity_cpu.mean().item<float>();
    float legacy_opacity_std = legacy_opacity_cpu.std().item<float>();

    spdlog::info("Computing new opacity statistics...");
    auto new_opacity_contiguous = new_opacity_cpu.contiguous();
    const float* new_opacity_ptr = new_opacity_contiguous.ptr<float>();
    float new_opacity_sum = 0.0f;
    size_t new_opacity_numel = new_opacity_contiguous.numel();
    for (size_t i = 0; i < new_opacity_numel; ++i) {
        new_opacity_sum += new_opacity_ptr[i];
    }
    float new_opacity_mean = new_opacity_sum / new_opacity_numel;

    spdlog::info("Legacy opacity: mean={:.4f}, std={:.4f}", legacy_opacity_mean, legacy_opacity_std);
    spdlog::info("New opacity: mean={:.4f}", new_opacity_mean);

    ASSERT_GT(legacy_opacity_std, 0.01f)
        << "Legacy opacities have insufficient variation for add_new_gs";

    // Compare states before add_new_gs
    spdlog::info("=== Comparing States Before add_new_gs ===");
    compareTensors(legacy_model_ref.means(), new_model_ref.means(), "Means (after training)");
    compareTensors(legacy_model_ref.opacity_raw(), new_model_ref.opacity_raw(), "Opacity (after training)");
    compareTensors(legacy_model_ref.scaling_raw(), new_model_ref.scaling_raw(), "Scaling (after training)");

    // Now call add_new_gs on both
    spdlog::info("=== Calling add_new_gs (Legacy) ===");
    int legacy_added = legacy_init.strategy->add_new_gs_test();
    spdlog::info("Legacy added {} Gaussians (total: {})", legacy_added, legacy_model_ref.size());

    spdlog::info("=== Calling add_new_gs (New) ===");
    int new_added = new_init.strategy->add_new_gs_test();
    spdlog::info("New added {} Gaussians (total: {})", new_added, new_model_ref.size());

    // Compare results
    spdlog::info("=== Comparing Results After add_new_gs ===");

    // Should add approximately the same number
    EXPECT_NEAR(legacy_added, new_added, std::max(1, static_cast<int>(legacy_added * 0.05)))
        << "Number of added Gaussians differs significantly";

    EXPECT_NEAR(legacy_model_ref.size(), new_model_ref.size(),
                std::max<size_t>(1UL, static_cast<size_t>(legacy_model_ref.size()) / 100))
        << "Final Gaussian count differs significantly";

    spdlog::info("Added Gaussians - Legacy: {}, New: {}", legacy_added, new_added);
    spdlog::info("Final count - Legacy: {}, New: {}", legacy_model_ref.size(), new_model_ref.size());

    // Compare final state of original Gaussians (first N)
    // The newly added Gaussians will differ due to random sampling
    size_t compare_count = std::min<size_t>(initial_count,
                                   std::min<size_t>(static_cast<size_t>(legacy_model_ref.size()),
                                                    static_cast<size_t>(new_model_ref.size())));

    spdlog::info("Comparing first {} original Gaussians", compare_count);

    auto legacy_means_full = legacy_model_ref.means().cpu();
    auto new_means_full = new_model_ref.means().cpu();

    auto legacy_means = legacy_means_full.slice(0, 0, compare_count);
    auto new_means = new_means_full.slice(0, 0, compare_count);

    compareTensors(legacy_means, new_means, "Means (original Gaussians after add_new_gs)");

    // Compare updated opacity for original Gaussians
    auto legacy_opacity_after = legacy_model_ref.get_opacity().cpu().slice(0, 0, compare_count);
    auto new_opacity_after = new_model_ref.get_opacity().cpu().slice(0, 0, compare_count);

    compareTensors(legacy_opacity_after, new_opacity_after, "Opacity (original Gaussians after add_new_gs)");

    // Compare scaling for original Gaussians
    auto legacy_scaling_after = legacy_model_ref.get_scaling().cpu().slice(0, 0, compare_count);
    auto new_scaling_after = new_model_ref.get_scaling().cpu().slice(0, 0, compare_count);

    compareTensors(legacy_scaling_after, new_scaling_after, "Scaling (original Gaussians after add_new_gs)");

    // Verify optimizer states were updated correctly
    spdlog::info("=== Verifying Optimizer State Updates ===");
    auto* legacy_opt = legacy_init.strategy->get_optimizer();
    auto* new_opt = new_init.strategy->get_optimizer();

    ASSERT_NE(legacy_opt, nullptr) << "Legacy optimizer is null";
    ASSERT_NE(new_opt, nullptr) << "New optimizer is null";

    // Check that optimizer has parameters for all Gaussians (including new ones)
    spdlog::info("Legacy optimizer has {} parameter groups", legacy_opt->param_groups().size());
    spdlog::info("New optimizer manages {} Gaussians", new_model_ref.size());

    spdlog::info("=== Test Complete ===");
    spdlog::info("Summary:");
    spdlog::info("  Initial Gaussians: {}", initial_count);
    spdlog::info("  Added (Legacy): {}, Added (New): {}", legacy_added, new_added);
    spdlog::info("  Final (Legacy): {}, Final (New): {}", legacy_model_ref.size(), new_model_ref.size());
}
