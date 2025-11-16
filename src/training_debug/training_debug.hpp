// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <expected>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <unordered_map>

// Include full definitions needed for struct members
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core_new/parameters.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"

// Forward declarations for camera types
namespace gs {
    class Camera;  // Legacy gs::Camera
}

namespace lfs::core {
    class Camera;  // New lfs::core::Camera
}

namespace gs::training {
    class CameraDataset;
    class MCMC;
}

namespace lfs::training {
    class CameraDataset;
    class MCMC;
}

namespace gs::training_debug {

/**
 * @brief Result of legacy (gs::) initialization - contains all data needed for training comparison
 */
struct LegacyInitializationResult {
    std::shared_ptr<gs::training::CameraDataset> dataset;
    std::shared_ptr<gs::SplatData> model;  // Shared pointer to Gaussian model
    std::shared_ptr<gs::training::MCMC> strategy;  // MCMC strategy instance with optimizer
    gs::param::TrainingParameters params;
    torch::Tensor background;  // Black background [0, 0, 0]
    torch::Tensor scene_center;
    std::unordered_map<size_t, std::shared_ptr<gs::Camera>> cam_id_to_cam;
    size_t num_gaussians;
};

/**
 * @brief Result of new (lfs::) initialization - contains all data needed for training comparison
 */
struct NewInitializationResult {
    std::shared_ptr<lfs::training::CameraDataset> dataset;
    std::shared_ptr<lfs::core::SplatData> model;  // Shared pointer to Gaussian model
    std::shared_ptr<lfs::training::MCMC> strategy;  // MCMC strategy instance with optimizer
    lfs::core::param::TrainingParameters params;
    lfs::core::Tensor background;  // Black background [0, 0, 0]
    lfs::core::Tensor scene_center;
    std::unordered_map<size_t, std::shared_ptr<lfs::core::Camera>> cam_id_to_cam;
    size_t num_gaussians;
};

/**
 * @brief Load dataset using the legacy gs::loader module
 *
 * Uses the old LibTorch-based loader from src/training/
 *
 * @return Success message on success, error string on failure
 */
std::expected<void, std::string> load_dataset_legacy();

/**
 * @brief Load dataset using the new lfs::loader module
 *
 * Uses the new LibTorch-free loader from src/training_new/
 *
 * @return Success message on success, error string on failure
 */
std::expected<void, std::string> load_dataset_new();

/**
 * @brief Initialize COLMAP dataset with MCMC strategy using legacy gs:: modules
 *
 * Loads COLMAP dataset, initializes from point cloud, creates MCMC strategy,
 * and returns all initialization data for step-by-step debugging comparison.
 *
 * @return LegacyInitializationResult on success, error string on failure
 */
std::expected<LegacyInitializationResult, std::string> initialize_legacy();

/**
 * @brief Initialize COLMAP dataset with MCMC strategy using new lfs:: modules
 *
 * Loads COLMAP dataset, initializes from point cloud, creates MCMC strategy,
 * and returns all initialization data for step-by-step debugging comparison.
 *
 * @return NewInitializationResult on success, error string on failure
 */
std::expected<NewInitializationResult, std::string> initialize_new();

/**
 * @brief Initialize and compare both legacy and new training pipelines
 *
 * Convenience function that:
 * 1. Calls initialize_legacy()
 * 2. Calls initialize_new()
 * 3. Returns both results for direct comparison
 *
 * @return Pair of (LegacyInitializationResult, NewInitializationResult) on success, error string on failure
 */
std::expected<std::pair<LegacyInitializationResult, NewInitializationResult>, std::string> initialize_both();

/**
 * @brief Render and save comparison images for both pipelines
 *
 * Takes the initialization results from both pipelines, picks the same camera from each dataset,
 * renders it using fast_rasterize_forward, and saves a comparison image (1x2 grid + GT).
 *
 * @param legacy_init Legacy initialization result
 * @param new_init New initialization result
 * @param camera_index Index of camera to render (default: 0)
 * @param output_path Path to save the comparison image (default: "render_comparison.png")
 * @return Success message on success, error string on failure
 */
std::expected<void, std::string> render_and_save_comparison(
    LegacyInitializationResult& legacy_init,
    NewInitializationResult& new_init,
    size_t camera_index = 0,
    const std::string& output_path = "render_comparison.png");

/**
 * @brief Run training loop comparison for both pipelines
 *
 * Runs a training loop on a single camera for both legacy and new pipelines,
 * comparing:
 * - Photometric loss values
 * - Gradient values after backward pass
 * - Optimizer betas and state
 * - Gaussian attribute values after optimizer step
 *
 * @param legacy_init Legacy initialization result with MCMC strategy
 * @param new_init New initialization result with MCMC strategy
 * @param camera_index Index of camera to use for training (default: 0)
 * @param max_iterations Maximum number of training iterations (default: 10)
 * @return Success on success, error string on failure
 */
std::expected<void, std::string> run_training_loop_comparison(
    LegacyInitializationResult& legacy_init,
    NewInitializationResult& new_init,
    size_t camera_index = 0,
    int max_iterations = 10);

} // namespace gs::training_debug
