// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "training_debug.hpp"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <format>
#include <variant>

// Legacy modules (LibTorch-based)
#include "loader/loader.hpp"
#include "loader/cache_image_loader.hpp"
#include "core/splat_data.hpp"
#include "core/point_cloud.hpp"
#include "core/parameters.hpp"
#include "core/camera.hpp"
#include "optimizers/fused_adam.hpp"

// New modules (LibTorch-free)
#include "loader_new/loader.hpp"
#include "loader_new/cache_image_loader.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/point_cloud.hpp"
#include "core_new/parameters.hpp"
#include "core_new/camera.hpp"

// NOTE: DO NOT include core/logger.hpp or core_new/logger.hpp here!
// Both define the same global macros (LOG_INFO, LOG_DEBUG, etc.) causing conflicts.
// Use spdlog:: functions directly instead.

// Include dataset headers AFTER namespace-specific includes to avoid conflicts
#include "../training/dataset.hpp"           // gs::training::CameraDataset
#include "../training_new/dataset.hpp"       // lfs::training::CameraDataset

// Include strategy headers AFTER datasets
#include "../training/strategies/mcmc.hpp"       // gs::training::MCMC
#include "../training_new/strategies/mcmc.hpp"   // lfs::training::MCMC

// Include render output structs
#include "../training/rasterization/rasterizer.hpp"       // gs::training::RenderOutput
#include "../training_new/optimizer/render_output.hpp"    // lfs::training::RenderOutput

// IMPORTANT: Include NEW rasterization_api.h BEFORE fast_rasterizer.hpp
// Must use full path because there are multiple rasterization_api.h files!
#include "../training_new/fastgs/rasterization/include/rasterization_api.h"   // Defines fast_lfs::rasterization namespace

// Include fast_rasterizer headers for both pipelines
#include "../training/rasterization/fast_rasterizer.hpp"       // gs::training::fast_rasterize_forward
#include "../training_new/rasterization/fast_rasterizer.hpp"   // lfs::training::fast_rasterize_forward

// Include losses for training loop comparison (new implementation)
#include "../training_new/losses/photometric_loss.hpp"         // lfs::training::losses::PhotometricLoss for lfs::core::Tensor
#include "kernels/fused_ssim.cuh"                               // fused_ssim for legacy loss computation

// Include image I/O for saving (now properly namespaced)
#include "core/image_io.hpp"
#include "core_new/image_io.hpp"

namespace gs::training_debug {

std::expected<void, std::string> load_dataset_legacy() {
    spdlog::info("=== Loading dataset using LEGACY gs::loader ===");

    // 1. Create loader
    auto loader = gs::loader::Loader::create();
    if (!loader) {
        return std::unexpected("Failed to create legacy loader");
    }

    // 2. Set up load options
    gs::loader::LoadOptions load_options{
        .resize_factor = -1,
        .max_width = 3840,
        .images_folder = "images",
        .validate_only = false,
        .progress = [](float percentage, const std::string& message) {
            spdlog::debug("[LEGACY] [{:5.1f}%] {}", percentage, message);
        }
    };

    // 3. Load the dataset
    std::filesystem::path data_path = "/media/paja/T7/my_data/fasnacht";
    spdlog::info("[LEGACY] Loading dataset from: {}", data_path.string());

    auto load_result = loader->load(data_path, load_options);
    if (!load_result) {
        return std::unexpected("Legacy loader failed: " + load_result.error());
    }

    // 4. Report success
    spdlog::info("[LEGACY] Successfully loaded dataset");
    spdlog::info("[LEGACY] Loader used: {}", load_result->loader_used);
    spdlog::info("[LEGACY] Load time: {}ms", load_result->load_time.count());
    spdlog::info("[LEGACY] Warnings: {}", load_result->warnings.size());

    for (const auto& warning : load_result->warnings) {
        spdlog::warn("[LEGACY] {}", warning);
    }

    return {};
}

std::expected<void, std::string> load_dataset_new() {
    spdlog::info("=== Loading dataset using NEW lfs::loader ===");

    // 1. Create loader
    auto loader = lfs::loader::Loader::create();
    if (!loader) {
        return std::unexpected("Failed to create new loader");
    }

    // 2. Set up load options
    lfs::loader::LoadOptions load_options{
        .resize_factor = -1,
        .max_width = 3840,
        .images_folder = "images",
        .validate_only = false,
        .progress = [](float percentage, const std::string& message) {
            spdlog::debug("[NEW] [{:5.1f}%] {}", percentage, message);
        }
    };

    // 3. Load the dataset
    std::filesystem::path data_path = "/media/paja/T7/my_data/fasnacht";
    spdlog::info("[NEW] Loading dataset from: {}", data_path.string());

    auto load_result = loader->load(data_path, load_options);
    if (!load_result) {
        return std::unexpected("New loader failed: " + load_result.error());
    }

    // 4. Report success
    spdlog::info("[NEW] Successfully loaded dataset");
    spdlog::info("[NEW] Loader used: {}", load_result->loader_used);
    spdlog::info("[NEW] Load time: {}ms", load_result->load_time.count());
    spdlog::info("[NEW] Warnings: {}", load_result->warnings.size());

    for (const auto& warning : load_result->warnings) {
        spdlog::warn("[NEW] {}", warning);
    }

    return {};
}

std::expected<LegacyInitializationResult, std::string> initialize_legacy() {
    spdlog::info("=== Initializing LEGACY gs::training with MCMC strategy ===");

    // 1. Create minimal parameters for MCMC strategy
    gs::param::TrainingParameters params;
    params.dataset.data_path = "/media/paja/T7/my_data/garden";
    params.dataset.images = "images_4";
    params.dataset.resize_factor = -1;
    params.dataset.max_width = 3840;
    params.optimization.strategy = "mcmc";
    params.optimization.max_cap = 1000000;

    // 2. Create loader
    auto loader = gs::loader::Loader::create();
    if (!loader) {
        return std::unexpected("[LEGACY] Failed to create loader");
    }

    // 3. Initialize CacheLoader (required for image loading in legacy pipeline)
    auto& legacy_cache_loader = gs::loader::CacheLoader::getInstance(
        params.dataset.loading_params.use_cpu_memory,
        params.dataset.loading_params.use_fs_cache
    );
    spdlog::info("[LEGACY] CacheLoader initialized (cpu_memory={}, fs_cache={})",
                 params.dataset.loading_params.use_cpu_memory,
                 params.dataset.loading_params.use_fs_cache);

    // 4. Set up load options
    gs::loader::LoadOptions load_options{
        .resize_factor = params.dataset.resize_factor,
        .max_width = params.dataset.max_width,
        .images_folder = params.dataset.images,
        .validate_only = false,
        .progress = [](float percentage, const std::string& message) {
            spdlog::debug("[LEGACY] [{:5.1f}%] {}", percentage, message);
        }
    };

    // 5. Load the dataset
    spdlog::info("[LEGACY] Loading dataset from: {}", params.dataset.data_path.string());
    auto load_result = loader->load(params.dataset.data_path, load_options);
    if (!load_result) {
        return std::unexpected(std::format("[LEGACY] Failed to load dataset: {}", load_result.error()));
    }

    spdlog::info("[LEGACY] Dataset loaded successfully using {} loader", load_result->loader_used);

    // 5. Handle the loaded data and initialize
    return std::visit([&params, &load_result, &legacy_cache_loader](auto&& data) -> std::expected<LegacyInitializationResult, std::string> {
        using T = std::decay_t<decltype(data)>;

        if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatData>>) {
            return std::unexpected("[LEGACY] Direct PLY loading is not supported. Need COLMAP dataset.");
        } else if constexpr (std::is_same_v<T, gs::loader::LoadedScene>) {
            // Get point cloud
            gs::PointCloud point_cloud_to_use;
            if (data.point_cloud && data.point_cloud->size() > 0) {
                point_cloud_to_use = *data.point_cloud;
                spdlog::info("[LEGACY] Using point cloud with {} points", point_cloud_to_use.size());
            } else {
                return std::unexpected("[LEGACY] No point cloud provided");
            }

            // Initialize model from point cloud
            auto splat_result = gs::SplatData::init_model_from_pointcloud(
                params,
                load_result->scene_center,
                point_cloud_to_use);

            if (!splat_result) {
                return std::unexpected(
                    std::format("[LEGACY] Failed to initialize model: {}", splat_result.error()));
            }

            size_t num_gaussians = splat_result->size();
            spdlog::info("[LEGACY] Model initialized with {} Gaussians", num_gaussians);

            // Create MCMC strategy with the model (strategy takes ownership)
            auto strategy = std::make_shared<gs::training::MCMC>(std::move(*splat_result));
            spdlog::info("[LEGACY] MCMC strategy created with model");

            // Set active SH degree to max (3) for testing
            strategy->get_model().set_active_sh_degree(params.optimization.sh_degree);
            spdlog::info("[LEGACY] Set active SH degree to {}", strategy->get_model().get_active_sh_degree());

            // Initialize the strategy's optimizer
            strategy->initialize(params.optimization);
            spdlog::info("[LEGACY] Optimizer initialized");

            // Create background tensor (black background)
            torch::Tensor background = torch::tensor({0.f, 0.f, 0.f},
                                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            // Build camera cache
            std::unordered_map<size_t, std::shared_ptr<gs::Camera>> cam_id_to_cam;
            for (const auto& cam : data.cameras->get_cameras()) {
                cam_id_to_cam[cam->uid()] = cam;
            }
            spdlog::debug("[LEGACY] Camera cache initialized with {} cameras", cam_id_to_cam.size());

            // Update CacheLoader with dataset size
            legacy_cache_loader.update_cache_params(
                params.dataset.loading_params.use_cpu_memory,
                params.dataset.loading_params.use_fs_cache,
                static_cast<int>(data.cameras->size().value())
            );
            spdlog::debug("[LEGACY] Updated CacheLoader with {} expected images", data.cameras->size().value());

            // Report success
            spdlog::info("[LEGACY] ✓ Initialization complete");
            spdlog::info("[LEGACY]   - Dataset: {} cameras", data.cameras->size().value());
            spdlog::info("[LEGACY]   - Point cloud: {} points", point_cloud_to_use.size());
            spdlog::info("[LEGACY]   - Gaussians: {}", num_gaussians);
            spdlog::info("[LEGACY]   - Strategy: MCMC (initialized)");

            // Return all initialization data
            // Note: model pointer is nullptr since strategy owns the model now
            return LegacyInitializationResult{
                .dataset = data.cameras,
                .model = nullptr,  // Strategy owns the model
                .strategy = strategy,
                .params = params,
                .background = std::move(background),
                .scene_center = load_result->scene_center,
                .cam_id_to_cam = std::move(cam_id_to_cam),
                .num_gaussians = num_gaussians
            };
        } else {
            return std::unexpected("[LEGACY] Unknown data type returned from loader");
        }
    }, load_result->data);
}

std::expected<NewInitializationResult, std::string> initialize_new() {
    spdlog::info("=== Initializing NEW lfs::training with MCMC strategy ===");

    // 1. Create minimal parameters for MCMC strategy
    lfs::core::param::TrainingParameters params;
    params.dataset.data_path = "/media/paja/T7/my_data/garden";
    params.dataset.images = "images_4";
    params.dataset.resize_factor = -1;
    params.dataset.max_width = 3840;
    params.optimization.strategy = "mcmc";
    params.optimization.max_cap = 1000000;

    // 2. Create loader
    auto loader = lfs::loader::Loader::create();
    if (!loader) {
        return std::unexpected("[NEW] Failed to create loader");
    }

    // 3. Set up load options
    lfs::loader::LoadOptions load_options{
        .resize_factor = params.dataset.resize_factor,
        .max_width = params.dataset.max_width,
        .images_folder = params.dataset.images,
        .validate_only = false,
        .progress = [](float percentage, const std::string& message) {
            spdlog::debug("[NEW] [{:5.1f}%] {}", percentage, message);
        }
    };

    // 4. Initialize CacheLoader (required for image loading in new pipeline)
    // Note: CacheLoader is a singleton that needs to be initialized before loading images
    auto& cache_loader = lfs::loader::CacheLoader::getInstance(
        params.dataset.loading_params.use_cpu_memory,
        params.dataset.loading_params.use_fs_cache
    );
    spdlog::info("[NEW] CacheLoader initialized (cpu_memory={}, fs_cache={})",
                 params.dataset.loading_params.use_cpu_memory,
                 params.dataset.loading_params.use_fs_cache);

    // 5. Load the dataset
    spdlog::info("[NEW] Loading dataset from: {}", params.dataset.data_path.string());
    auto load_result = loader->load(params.dataset.data_path, load_options);
    if (!load_result) {
        return std::unexpected(std::format("[NEW] Failed to load dataset: {}", load_result.error()));
    }

    spdlog::info("[NEW] Dataset loaded successfully using {} loader", load_result->loader_used);

    // 5. Handle the loaded data and initialize
    return std::visit([&params, &load_result, &cache_loader](auto&& data) -> std::expected<NewInitializationResult, std::string> {
        using T = std::decay_t<decltype(data)>;

        if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
            return std::unexpected("[NEW] Direct PLY loading is not supported. Need COLMAP dataset.");
        } else if constexpr (std::is_same_v<T, lfs::loader::LoadedScene>) {
            // Get point cloud
            lfs::core::PointCloud point_cloud_to_use;
            if (data.point_cloud && data.point_cloud->size() > 0) {
                point_cloud_to_use = *data.point_cloud;
                spdlog::info("[NEW] Using point cloud with {} points", point_cloud_to_use.size());
            } else {
                return std::unexpected("[NEW] No point cloud provided");
            }

            // Initialize model from point cloud
            auto splat_result = lfs::core::SplatData::init_model_from_pointcloud(
                params,
                load_result->scene_center,
                point_cloud_to_use);

            if (!splat_result) {
                return std::unexpected(
                    std::format("[NEW] Failed to initialize model: {}", splat_result.error()));
            }

            size_t num_gaussians = splat_result->size();
            spdlog::info("[NEW] Model initialized with {} Gaussians", num_gaussians);

            // Create MCMC strategy with the model (strategy takes ownership)
            auto strategy = std::make_shared<lfs::training::MCMC>(std::move(*splat_result));
            spdlog::info("[NEW] MCMC strategy created with model");

            // Set active SH degree to max (3) for testing
            strategy->get_model().set_active_sh_degree(params.optimization.sh_degree);
            spdlog::info("[NEW] Set active SH degree to {}", strategy->get_model().get_active_sh_degree());

            // Initialize the strategy's optimizer
            strategy->initialize(params.optimization);
            spdlog::info("[NEW] Optimizer initialized");

            // Create background tensor (black background)
            lfs::core::Tensor background = lfs::core::Tensor::zeros({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

            // Build camera cache
            std::unordered_map<size_t, std::shared_ptr<lfs::core::Camera>> cam_id_to_cam;
            for (const auto& cam : data.cameras->get_cameras()) {
                cam_id_to_cam[cam->uid()] = cam;
            }
            spdlog::debug("[NEW] Camera cache initialized with {} cameras", cam_id_to_cam.size());

            // Update CacheLoader with dataset size
            cache_loader.update_cache_params(
                params.dataset.loading_params.use_cpu_memory,
                params.dataset.loading_params.use_fs_cache,
                static_cast<int>(data.cameras->size())
            );
            spdlog::debug("[NEW] Updated CacheLoader with {} expected images", data.cameras->size());

            // Report success
            spdlog::info("[NEW] ✓ Initialization complete");
            spdlog::info("[NEW]   - Dataset: {} cameras", data.cameras->size());
            spdlog::info("[NEW]   - Point cloud: {} points", point_cloud_to_use.size());
            spdlog::info("[NEW]   - Gaussians: {}", num_gaussians);
            spdlog::info("[NEW]   - Strategy: MCMC (initialized)");

            // Return all initialization data
            // Note: model pointer is nullptr since strategy owns the model now
            return NewInitializationResult{
                .dataset = data.cameras,
                .model = nullptr,  // Strategy owns the model
                .strategy = strategy,
                .params = params,
                .background = std::move(background),
                .scene_center = load_result->scene_center,
                .cam_id_to_cam = std::move(cam_id_to_cam),
                .num_gaussians = num_gaussians
            };
        } else {
            return std::unexpected("[NEW] Unknown data type returned from loader");
        }
    }, load_result->data);
}

std::expected<std::pair<LegacyInitializationResult, NewInitializationResult>, std::string> initialize_both() {
    spdlog::info("=== Initializing BOTH legacy and new training pipelines ===");

    // Initialize legacy pipeline
    auto legacy_result = initialize_legacy();
    if (!legacy_result) {
        return std::unexpected(std::format("Legacy initialization failed: {}", legacy_result.error()));
    }

    spdlog::info("--- Legacy initialization complete, starting new initialization ---");

    // Initialize new pipeline
    auto new_result = initialize_new();
    if (!new_result) {
        return std::unexpected(std::format("New initialization failed: {}", new_result.error()));
    }

    // Report comparison summary
    spdlog::info("=== Both pipelines initialized successfully ===");
    spdlog::info("Comparison summary:");
    spdlog::info("  Legacy Gaussians: {} | New Gaussians: {}",
                 legacy_result->num_gaussians, new_result->num_gaussians);
    spdlog::info("  Legacy cameras: {} | New cameras: {}",
                 legacy_result->dataset->size().value(), new_result->dataset->size());

    return std::make_pair(std::move(*legacy_result), std::move(*new_result));
}

std::expected<void, std::string> render_and_save_comparison(
    LegacyInitializationResult& legacy_init,
    NewInitializationResult& new_init,
    size_t camera_index,
    const std::string& output_path) {

    spdlog::info("=== Rendering comparison for camera {} ===", camera_index);

    // Validate camera index
    if (camera_index >= legacy_init.dataset->size().value()) {
        return std::unexpected(std::format("Camera index {} out of range (legacy has {} cameras)",
                                          camera_index, legacy_init.dataset->size().value()));
    }
    if (camera_index >= new_init.dataset->size()) {
        return std::unexpected(std::format("Camera index {} out of range (new has {} cameras)",
                                          camera_index, new_init.dataset->size()));
    }

    // Get the same camera from both datasets
    auto legacy_cameras = legacy_init.dataset->get_cameras();
    auto new_cameras = new_init.dataset->get_cameras();

    auto legacy_cam = legacy_cameras[camera_index];
    auto new_cam = new_cameras[camera_index];

    spdlog::info("[LEGACY] Rendering camera UID: {}", legacy_cam->uid());
    spdlog::info("[LEGACY] Camera dimensions: {}x{}", legacy_cam->image_width(), legacy_cam->image_height());
    spdlog::info("[LEGACY] Model has {} Gaussians", legacy_init.strategy->get_model().size());

    spdlog::info("[NEW] Rendering camera UID: {}", new_cam->uid());
    spdlog::info("[NEW] Camera dimensions: {}x{}", new_cam->image_width(), new_cam->image_height());
    spdlog::info("[NEW] Model has {} Gaussians", new_init.strategy->get_model().size());

    // Render using legacy pipeline
    spdlog::info("[LEGACY] Calling fast_rasterize...");
    auto legacy_output = gs::training::fast_rasterize(
        *legacy_cam,
        legacy_init.strategy->get_model(),
        legacy_init.background);

    spdlog::info("[LEGACY] Returned from fast_rasterize");
    spdlog::info("[LEGACY] RenderOutput.width = {}, RenderOutput.height = {}",
                 legacy_output.width, legacy_output.height);

    // Render using new pipeline
    auto new_result = lfs::training::fast_rasterize_forward(
        *new_cam,
        new_init.strategy->get_model(),
        new_init.background);

    spdlog::info("[NEW] Rendered {}x{} image", new_result.first.width, new_result.first.height);

    // Get ground truth images
    auto legacy_gt = legacy_cam->load_and_get_image(legacy_init.params.dataset.resize_factor,
                                                      legacy_init.params.dataset.max_width);
    auto new_gt = new_cam->load_and_get_image(new_init.params.dataset.resize_factor,
                                               new_init.params.dataset.max_width);

    // Compare GT images
    spdlog::info("[GT COMPARISON] Legacy GT shape: [{}, {}, {}]",
                 legacy_gt.size(0), legacy_gt.size(1), legacy_gt.size(2));
    spdlog::info("[GT COMPARISON] New GT shape: [{}, {}, {}]",
                 new_gt.shape()[0], new_gt.shape()[1], new_gt.shape()[2]);

    // Convert new GT to torch tensor for comparison
    auto new_gt_cpu = new_gt.to(lfs::core::Device::CPU);
    torch::Tensor new_gt_torch = torch::from_blob(
        new_gt_cpu.ptr<float>(),
        {static_cast<long>(new_gt_cpu.shape()[0]),
         static_cast<long>(new_gt_cpu.shape()[1]),
         static_cast<long>(new_gt_cpu.shape()[2])},
        torch::kFloat32).clone();

    // Compare values
    auto legacy_gt_cpu = legacy_gt.cpu();
    float max_diff = (legacy_gt_cpu - new_gt_torch).abs().max().template item<float>();
    float mean_diff = (legacy_gt_cpu - new_gt_torch).abs().mean().template item<float>();

    spdlog::info("[GT COMPARISON] Max difference: {}", max_diff);
    spdlog::info("[GT COMPARISON] Mean difference: {}", mean_diff);

    if (max_diff > 1e-5) {
        spdlog::warn("[GT COMPARISON] GT images differ significantly!");
        // Sample some values for debugging
        spdlog::info("[GT COMPARISON] Legacy GT [0,0,0] = {}", legacy_gt_cpu[0][0][0].template item<float>());
        spdlog::info("[GT COMPARISON] New GT [0,0,0] = {}", new_gt_torch[0][0][0].template item<float>());
        spdlog::info("[GT COMPARISON] Legacy GT [0,0,1] = {}", legacy_gt_cpu[0][0][1].template item<float>());
        spdlog::info("[GT COMPARISON] New GT [0,0,1] = {}", new_gt_torch[0][0][1].template item<float>());
        spdlog::info("[GT COMPARISON] Legacy GT [1,0,0] = {}", legacy_gt_cpu[1][0][0].template item<float>());
        spdlog::info("[GT COMPARISON] New GT [1,0,0] = {}", new_gt_torch[1][0][0].template item<float>());
        spdlog::info("[GT COMPARISON] Legacy GT [0,100,100] = {}", legacy_gt_cpu[0][100][100].template item<float>());
        spdlog::info("[GT COMPARISON] New GT [0,100,100] = {}", new_gt_torch[0][100][100].template item<float>());

        // Check tensor strides
        spdlog::info("[GT COMPARISON] Legacy GT strides: [{}, {}, {}]",
                     legacy_gt_cpu.stride(0), legacy_gt_cpu.stride(1), legacy_gt_cpu.stride(2));
        spdlog::info("[GT COMPARISON] New GT strides: [{}, {}, {}]",
                     new_gt.stride(0), new_gt.stride(1), new_gt.stride(2));
    } else {
        spdlog::info("[GT COMPARISON] GT images match!");
    }

    // Save legacy comparison (rendered + GT) - horizontal 1x2 grid
    std::filesystem::path legacy_path = std::filesystem::path(output_path).parent_path() /
                                        (std::filesystem::path(output_path).stem().string() + "_legacy.png");
    image_io::BatchImageSaver::instance().queue_save_multiple(legacy_path, {legacy_output.image, legacy_gt}, true, 5);
    spdlog::info("[LEGACY] Saved comparison to: {}", legacy_path.string());

    // For new pipeline, save side-by-side comparison (rendered + GT) - horizontal 1x2 grid
    std::filesystem::path new_path = std::filesystem::path(output_path).parent_path() /
                                     (std::filesystem::path(output_path).stem().string() + "_new.png");

    spdlog::info("[NEW] Creating side-by-side comparison: rendered + GT");
    spdlog::info("[NEW] Rendered image shape: [{}, {}, {}]",
                 new_result.first.image.shape()[0],
                 new_result.first.image.shape()[1],
                 new_result.first.image.shape()[2]);
    spdlog::info("[NEW] GT image shape: [{}, {}, {}]",
                 new_gt.shape()[0],
                 new_gt.shape()[1],
                 new_gt.shape()[2]);

    // Save new pipeline comparison (rendered + GT) - horizontal 1x2 grid
    lfs::core::save_image(new_path, {new_result.first.image, new_gt}, true, 5);
    spdlog::info("[NEW] Saved comparison to: {}", new_path.string());

    spdlog::info("=== Rendering comparison complete ===");
    return {};
}

std::expected<void, std::string> run_training_loop_comparison(
    LegacyInitializationResult& legacy_init,
    NewInitializationResult& new_init,
    size_t camera_index,
    int max_iterations) {

    spdlog::info("=== Starting training loop comparison ===");
    spdlog::info("Camera index: {}, Max iterations: {}", camera_index, max_iterations);

    // Validate inputs
    if (camera_index >= legacy_init.dataset->size().value()) {
        return std::unexpected(std::format("Camera index {} out of range (legacy has {} cameras)",
                                          camera_index, legacy_init.dataset->size().value()));
    }
    if (camera_index >= new_init.dataset->size()) {
        return std::unexpected(std::format("Camera index {} out of range (new has {} cameras)",
                                          camera_index, new_init.dataset->size()));
    }

    // Get cameras
    auto legacy_cameras = legacy_init.dataset->get_cameras();
    auto new_cameras = new_init.dataset->get_cameras();
    auto legacy_cam = legacy_cameras[camera_index];
    auto new_cam = new_cameras[camera_index];

    // Load GT images once
    auto legacy_gt = legacy_cam->load_and_get_image(legacy_init.params.dataset.resize_factor,
                                                      legacy_init.params.dataset.max_width);
    auto new_gt = new_cam->load_and_get_image(new_init.params.dataset.resize_factor,
                                               new_init.params.dataset.max_width);

    spdlog::info("Loaded GT images - Legacy: [{}, {}, {}], New: [{}, {}, {}]",
                 legacy_gt.size(0), legacy_gt.size(1), legacy_gt.size(2),
                 new_gt.shape()[0], new_gt.shape()[1], new_gt.shape()[2]);

    // ============================================================
    // INITIAL PARAMETER COMPARISON (BEFORE ANY TRAINING)
    // ============================================================
    spdlog::info("");
    spdlog::info("=== INITIAL PARAMETER COMPARISON (Before Training) ===");

    {
        auto& legacy_model = legacy_init.strategy->get_model();
        auto& new_model = new_init.strategy->get_model();

        // Compare opacity_raw
        auto legacy_opacity = legacy_model.opacity_raw().cpu();
        auto new_opacity_cpu = new_model.opacity_raw().to(lfs::core::Device::CPU);
        torch::Tensor new_opacity_torch = torch::empty({static_cast<long>(new_opacity_cpu.shape()[0]),
                                                         static_cast<long>(new_opacity_cpu.shape()[1])}, torch::kFloat32);
        std::memcpy(new_opacity_torch.template data_ptr<float>(), new_opacity_cpu.ptr<float>(),
                    new_opacity_cpu.numel() * sizeof(float));

        auto opacity_diff = (legacy_opacity - new_opacity_torch).abs();
        spdlog::info("Opacity Raw Initial - Max diff: {:.6e}, Mean diff: {:.6e}",
                     opacity_diff.max().item().toFloat(),
                     opacity_diff.mean().item().toFloat());

        // Compare scaling_raw
        auto legacy_scaling = legacy_model.scaling_raw().cpu();
        auto new_scaling_cpu = new_model.scaling_raw().to(lfs::core::Device::CPU);
        torch::Tensor new_scaling_torch = torch::empty({static_cast<long>(new_scaling_cpu.shape()[0]),
                                                         static_cast<long>(new_scaling_cpu.shape()[1])}, torch::kFloat32);
        std::memcpy(new_scaling_torch.template data_ptr<float>(), new_scaling_cpu.ptr<float>(),
                    new_scaling_cpu.numel() * sizeof(float));

        auto scaling_diff = (legacy_scaling - new_scaling_torch).abs();
        spdlog::info("Scaling Raw Initial - Max diff: {:.6e}, Mean diff: {:.6e}",
                     scaling_diff.max().item().toFloat(),
                     scaling_diff.mean().item().toFloat());

        // Compare means
        auto legacy_means = legacy_model.means().cpu();
        auto new_means_cpu = new_model.means().to(lfs::core::Device::CPU);
        torch::Tensor new_means_torch = torch::empty({static_cast<long>(new_means_cpu.shape()[0]),
                                                       static_cast<long>(new_means_cpu.shape()[1])}, torch::kFloat32);
        std::memcpy(new_means_torch.template data_ptr<float>(), new_means_cpu.ptr<float>(),
                    new_means_cpu.numel() * sizeof(float));

        auto means_diff = (legacy_means - new_means_torch).abs();
        spdlog::info("Means Initial - Max diff: {:.6e}, Mean diff: {:.6e}",
                     means_diff.max().item().toFloat(),
                     means_diff.mean().item().toFloat());
    }

    // Training loop
    for (int iter = 1; iter <= max_iterations; ++iter) {
        spdlog::info("");
        spdlog::info("=== Iteration {} / {} ===", iter, max_iterations);

        // ============================================================
        // 0.5. COMPARE RASTERIZER INPUTS (Iteration 1 only)
        // ============================================================
        if (iter == 1) {
            spdlog::info("[{}] === Rasterizer Input Comparison ===", iter);

            auto& legacy_model = legacy_init.strategy->get_model();
            auto& new_model = new_init.strategy->get_model();

            // Compare active SH degree
            spdlog::info("[{}] Active SH Degree - Legacy: {}, New: {}",
                         iter, legacy_model.get_active_sh_degree(), new_model.get_active_sh_degree());

            // Log the activated values shape
            spdlog::info("[{}] Model sizes - Legacy: {}, New: {}", iter, legacy_model.size(), new_model.size());
        }

        // ============================================================
        // 1. FORWARD PASS - Render from both pipelines
        // ============================================================
        spdlog::info("[{}] === Forward Pass (Rendering) ===", iter);

        // Use strategy's model (which gets updated by optimizer)
        auto legacy_output = gs::training::fast_rasterize(
            *legacy_cam,
            legacy_init.strategy->get_model(),
            legacy_init.background);

        auto new_render_result = lfs::training::fast_rasterize_forward(
            *new_cam,
            new_init.strategy->get_model(),
            new_init.background);
        auto [new_output, new_ctx] = new_render_result;

        spdlog::info("[{}] Rendered - Legacy: {}x{}, New: {}x{}",
                     iter,
                     legacy_output.width, legacy_output.height,
                     new_output.width, new_output.height);

        // ============================================================
        // 1.5. COMPARE RENDERED IMAGES (iteration 1 only for detailed analysis)
        // ============================================================
        if (iter == 1) {
            spdlog::info("[{}] === Rendered Image Comparison ===", iter);

            try {
                // legacy_output.image is torch::Tensor [3, H, W] or [H, W, 3]
                // new_output.image is lfs::core::Tensor [3, H, W] or [H, W, 3]

                auto legacy_img = legacy_output.image.cpu();
                auto new_img_cpu = new_output.image.to(lfs::core::Device::CPU);

            // Convert new image to torch for comparison
            std::vector<long> img_shape_vec;
            for (size_t i = 0; i < new_img_cpu.ndim(); ++i) {
                img_shape_vec.push_back(static_cast<long>(new_img_cpu.shape()[i]));
            }
            torch::Tensor new_img_torch = torch::from_blob(
                const_cast<float*>(new_img_cpu.ptr<float>()),
                torch::IntArrayRef(img_shape_vec),
                torch::kFloat32).clone();

            spdlog::info("[{}] Legacy image shape: [{}, {}, {}]",
                        iter, legacy_img.size(0), legacy_img.size(1), legacy_img.size(2));
            spdlog::info("[{}] New image shape: [{}, {}, {}]",
                        iter, new_img_torch.size(0), new_img_torch.size(1), new_img_torch.size(2));

            // Compute differences
            auto img_diff = (legacy_img - new_img_torch).abs();
            float img_max_diff = img_diff.max().template item<float>();
            float img_mean_diff = img_diff.mean().template item<float>();

            spdlog::info("[{}] Rendered Image - Max diff: {:.6e}, Mean diff: {:.6e}",
                        iter, img_max_diff, img_mean_diff);

            // Per-channel differences
            if (legacy_img.size(0) == 3) {
                // [3, H, W] layout
                for (int c = 0; c < 3; ++c) {
                    float ch_max = img_diff[c].max().template item<float>();
                    float ch_mean = img_diff[c].mean().template item<float>();
                    const char* ch_name = (c == 0) ? "R" : (c == 1) ? "G" : "B";
                    spdlog::info("[{}]   Channel {} - Max diff: {:.6e}, Mean diff: {:.6e}",
                                iter, ch_name, ch_max, ch_mean);
                }

                // Show first few pixels
                spdlog::info("[{}] First 3 pixels (top-left corner):", iter);
                for (int i = 0; i < 3; ++i) {
                    auto legacy_pixel = legacy_img.index({torch::indexing::Slice(), 0, i});
                    auto new_pixel = new_img_torch.index({torch::indexing::Slice(), 0, i});
                    spdlog::info("[{}]   Pixel (0,{}): Legacy=[{:.6f}, {:.6f}, {:.6f}], New=[{:.6f}, {:.6f}, {:.6f}]",
                                iter, i,
                                legacy_pixel[0].template item<float>(),
                                legacy_pixel[1].template item<float>(),
                                legacy_pixel[2].template item<float>(),
                                new_pixel[0].template item<float>(),
                                new_pixel[1].template item<float>(),
                                new_pixel[2].template item<float>());
                }
            } else {
                // [H, W, 3] layout
                for (int c = 0; c < 3; ++c) {
                    auto ch_diff = img_diff.index({torch::indexing::Slice(), torch::indexing::Slice(), c});
                    float ch_max = ch_diff.max().template item<float>();
                    float ch_mean = ch_diff.mean().template item<float>();
                    const char* ch_name = (c == 0) ? "R" : (c == 1) ? "G" : "B";
                    spdlog::info("[{}]   Channel {} - Max diff: {:.6e}, Mean diff: {:.6e}",
                                iter, ch_name, ch_max, ch_mean);
                }

                // Show first few pixels
                spdlog::info("[{}] First 3 pixels (top-left corner):", iter);
                for (int i = 0; i < 3; ++i) {
                    auto legacy_pixel = legacy_img.index({0, i, torch::indexing::Slice()});
                    auto new_pixel = new_img_torch.index({0, i, torch::indexing::Slice()});
                    spdlog::info("[{}]   Pixel (0,{}): Legacy=[{:.6f}, {:.6f}, {:.6f}], New=[{:.6f}, {:.6f}, {:.6f}]",
                                iter, i,
                                legacy_pixel[0].template item<float>(),
                                legacy_pixel[1].template item<float>(),
                                legacy_pixel[2].template item<float>(),
                                new_pixel[0].template item<float>(),
                                new_pixel[1].template item<float>(),
                                new_pixel[2].template item<float>());
                }
            }
            } catch (const std::exception& e) {
                spdlog::error("[{}] Image comparison failed: {}", iter, e.what());
            }
        }

        // ============================================================
        // 2. COMPUTE PHOTOMETRIC LOSS
        // ============================================================
        spdlog::info("[{}] === Computing Photometric Loss ===", iter);

        // Legacy loss computation (inline from trainer.cpp)
        torch::Tensor legacy_rendered = legacy_output.image;
        legacy_rendered = legacy_rendered.dim() == 3 ? legacy_rendered.unsqueeze(0) : legacy_rendered;
        torch::Tensor legacy_gt_4d = legacy_gt.dim() == 3 ? legacy_gt.unsqueeze(0) : legacy_gt;

        auto legacy_l1_loss = torch::l1_loss(legacy_rendered, legacy_gt_4d);
        auto legacy_ssim_loss = 1.f - fused_ssim(legacy_rendered, legacy_gt_4d, "valid", /*train=*/true);
        torch::Tensor legacy_loss_tensor = (1.f - legacy_init.params.optimization.lambda_dssim) * legacy_l1_loss +
                                            legacy_init.params.optimization.lambda_dssim * legacy_ssim_loss;
        float legacy_loss_value = legacy_loss_tensor.template item<float>();

        // New loss
        lfs::training::losses::PhotometricLoss::Params new_loss_params{
            .lambda_dssim = new_init.params.optimization.lambda_dssim
        };
        auto new_loss_result = lfs::training::losses::PhotometricLoss::forward(
            new_output.image, new_gt, new_loss_params);
        if (!new_loss_result) {
            return std::unexpected(std::format("New loss computation failed: {}", new_loss_result.error()));
        }
        auto [new_loss_tensor, new_loss_ctx] = *new_loss_result;
        float new_loss_value = new_loss_tensor.template item<float>();  // Extract scalar from tensor

        spdlog::info("[{}] Loss - Legacy: {:.6f}, New: {:.6f}, Diff: {:.6f}",
                     iter,
                     legacy_loss_value,
                     new_loss_value,
                     std::abs(legacy_loss_value - new_loss_value));

        // ============================================================
        // 2.5. CHECK SH0 VALUES BEFORE BACKWARD (iteration 1 only for detailed analysis)
        // ============================================================
        if (iter == 1) {
            spdlog::info("[{}] === SH0 Memory Analysis BEFORE Backward ===", iter);

            auto& legacy_model_pre = legacy_init.strategy->get_model();
            auto& new_model_pre = new_init.strategy->get_model();

            // Legacy SH0
            auto legacy_sh0_pre = legacy_model_pre.sh0();
            const float* legacy_sh0_ptr = legacy_sh0_pre.template data_ptr<float>();

            // New SH0
            auto& new_sh0_pre = new_model_pre.sh0();
            const float* new_sh0_ptr = new_sh0_pre.ptr<float>();

            // Check first 3 Gaussians using tensor indexing (safer)
            spdlog::info("[{}] First 3 Gaussians - Values:", iter);
            auto legacy_sh0_cpu_check = legacy_sh0_pre.cpu().reshape({legacy_sh0_pre.size(0), 3});
            auto new_sh0_flat_check = new_sh0_pre.contiguous().flatten(1, 2);
            auto new_sh0_cpu_check = new_sh0_flat_check.to(lfs::core::Device::CPU);
            size_t N_check = new_sh0_cpu_check.shape()[0];
            torch::Tensor new_sh0_torch_check = torch::empty({static_cast<long>(N_check), 3}, torch::kFloat32);
            std::memcpy(new_sh0_torch_check.template data_ptr<float>(), new_sh0_cpu_check.ptr<float>(), N_check * 3 * sizeof(float));

            for (int i = 0; i < 3; ++i) {
                auto legacy_row = legacy_sh0_cpu_check[i];
                auto new_row = new_sh0_torch_check[i];
                spdlog::info("[{}]   Gaussian {}: Legacy=[{:.6f}, {:.6f}, {:.6f}], New=[{:.6f}, {:.6f}, {:.6f}]",
                           iter, i,
                           legacy_row[0].item().toFloat(), legacy_row[1].item().toFloat(), legacy_row[2].item().toFloat(),
                           new_row[0].item().toFloat(), new_row[1].item().toFloat(), new_row[2].item().toFloat());
            }

            // Check tensor properties
            spdlog::info("[{}] Legacy SH0: shape=[{}, {}, {}], is_contiguous={}",
                        iter, legacy_sh0_pre.size(0), legacy_sh0_pre.size(1), legacy_sh0_pre.size(2),
                        legacy_sh0_pre.is_contiguous());
            spdlog::info("[{}] New SH0: shape=[{}, {}, {}], is_contiguous={}",
                        iter, new_sh0_pre.shape()[0], new_sh0_pre.shape()[1], new_sh0_pre.shape()[2],
                        new_sh0_pre.is_contiguous());

            // Compute difference
            auto legacy_sh0_cpu = legacy_sh0_pre.cpu().reshape({legacy_sh0_pre.size(0), 3});
            auto new_sh0_flat = new_sh0_pre.contiguous().flatten(1, 2);
            auto new_sh0_cpu = new_sh0_flat.to(lfs::core::Device::CPU);
            size_t N = new_sh0_cpu.shape()[0];
            size_t C = new_sh0_cpu.shape()[1];
            torch::Tensor new_sh0_torch = torch::empty({static_cast<long>(N), static_cast<long>(C)}, torch::kFloat32);
            std::memcpy(new_sh0_torch.template data_ptr<float>(), new_sh0_cpu.ptr<float>(), N * C * sizeof(float));

            float sh0_diff_before = (legacy_sh0_cpu - new_sh0_torch).abs().max().item().toFloat();
            spdlog::info("[{}] SH0 max difference BEFORE backward: {:.6f}", iter, sh0_diff_before);
        }

        // ============================================================
        // 3. BACKWARD PASS - Compute gradients
        // ============================================================
        spdlog::info("[{}] === Backward Pass ===", iter);

        // Allocate gradients if needed (first iteration)
        if (!new_init.strategy->get_model().has_gradients()) {
            new_init.strategy->get_model().allocate_gradients();
            spdlog::info("[{}] Allocated gradients for new model", iter);
        } else {
            // Zero gradients before backward
            new_init.strategy->get_model().zero_gradients();
        }
        // Note: Legacy (LibTorch) manages gradients automatically

        // Legacy backward - PyTorch autograd handles everything automatically
        legacy_loss_tensor.backward();

        // New backward
        lfs::training::fast_rasterize_backward(new_ctx, new_loss_ctx.grad_image, new_init.strategy->get_model());

        spdlog::info("[{}] Backward complete", iter);

        // ============================================================
        // 4. COMPARE GRADIENTS
        // ============================================================
        spdlog::info("[{}] === Gradient Comparison ===", iter);

        // Compare means gradients
        auto legacy_means_grad = legacy_init.strategy->get_model().means().grad();
        auto new_means_grad = new_init.strategy->get_model().means_grad();

        // Log shapes with proper formatting
        if (legacy_means_grad.defined()) {
            auto sizes = legacy_means_grad.sizes();
            std::string shape_str = "[";
            for (size_t i = 0; i < sizes.size(); ++i) {
                if (i > 0) shape_str += ", ";
                shape_str += std::to_string(sizes[i]);
            }
            shape_str += "]";
            spdlog::info("[{}] Legacy means grad defined: true, shape: {}",
                         iter, shape_str);
        } else {
            spdlog::info("[{}] Legacy means grad defined: false", iter);
        }
        spdlog::info("[{}] New means grad shape: [{}, {}]",
                     iter, new_means_grad.shape()[0], new_means_grad.shape()[1]);

        // Convert new grad to torch for comparison
        auto new_means_grad_cpu = new_means_grad.to(lfs::core::Device::CPU);
        torch::Tensor new_means_grad_torch = torch::from_blob(
            new_means_grad_cpu.ptr<float>(),
            {static_cast<long>(new_means_grad_cpu.shape()[0]),
             static_cast<long>(new_means_grad_cpu.shape()[1])},
            torch::kFloat32).clone();

        if (legacy_means_grad.defined() && legacy_means_grad.numel() > 0) {
            auto legacy_means_grad_cpu = legacy_means_grad.cpu();
            float means_grad_max_diff = (legacy_means_grad_cpu - new_means_grad_torch).abs().max().template item<float>();
            float means_grad_mean_diff = (legacy_means_grad_cpu - new_means_grad_torch).abs().mean().template item<float>();

            spdlog::info("[{}] Means Gradient - Max diff: {:.2e}, Mean diff: {:.2e}",
                         iter, means_grad_max_diff, means_grad_mean_diff);
        } else {
            spdlog::warn("[{}] Legacy means gradient not defined or empty, skipping comparison", iter);
        }

        // ============================================================
        // 4.5. COMPARE OPACITY AND SCALING GRADIENTS (Iteration 1 detailed)
        // ============================================================
        if (iter <= 2) {  // Log first 2 iterations
            spdlog::info("[{}] === Opacity/Scaling Gradient Detailed Comparison ===", iter);

            // Opacity gradients
            auto legacy_opacity_grad = legacy_init.strategy->get_model().opacity_raw().grad();
            auto new_opacity_grad = new_init.strategy->get_model().opacity_grad();

            if (legacy_opacity_grad.defined() && legacy_opacity_grad.numel() > 0) {
                auto legacy_opacity_grad_cpu = legacy_opacity_grad.cpu();
                auto new_opacity_grad_cpu = new_opacity_grad.to(lfs::core::Device::CPU);
                torch::Tensor new_opacity_grad_torch = torch::from_blob(
                    new_opacity_grad_cpu.ptr<float>(),
                    {static_cast<long>(new_opacity_grad_cpu.shape()[0]),
                     static_cast<long>(new_opacity_grad_cpu.shape()[1])},
                    torch::kFloat32).clone();

                float opacity_grad_max_diff = (legacy_opacity_grad_cpu - new_opacity_grad_torch).abs().max().template item<float>();
                float opacity_grad_mean_diff = (legacy_opacity_grad_cpu - new_opacity_grad_torch).abs().mean().template item<float>();

                spdlog::info("[{}] Opacity Gradient - Max diff: {:.2e}, Mean diff: {:.2e}",
                             iter, opacity_grad_max_diff, opacity_grad_mean_diff);

                // Log first 5 gradients
                spdlog::info("[{}] First 5 Opacity Gradients:", iter);
                for (int i = 0; i < std::min(5, static_cast<int>(legacy_opacity_grad_cpu.size(0))); ++i) {
                    float legacy_val = legacy_opacity_grad_cpu[i][0].template item<float>();
                    float new_val = new_opacity_grad_torch[i][0].template item<float>();
                    spdlog::info("[{}]   Gaussian {}: Legacy={:.6e}, New={:.6e}, Diff={:.6e}",
                                 iter, i, legacy_val, new_val, std::abs(legacy_val - new_val));
                }
            }

            // Scaling gradients
            auto legacy_scaling_grad = legacy_init.strategy->get_model().scaling_raw().grad();
            auto new_scaling_grad = new_init.strategy->get_model().scaling_grad();

            if (legacy_scaling_grad.defined() && legacy_scaling_grad.numel() > 0) {
                auto legacy_scaling_grad_cpu = legacy_scaling_grad.cpu();
                auto new_scaling_grad_cpu = new_scaling_grad.to(lfs::core::Device::CPU);
                torch::Tensor new_scaling_grad_torch = torch::from_blob(
                    new_scaling_grad_cpu.ptr<float>(),
                    {static_cast<long>(new_scaling_grad_cpu.shape()[0]),
                     static_cast<long>(new_scaling_grad_cpu.shape()[1])},
                    torch::kFloat32).clone();

                float scaling_grad_max_diff = (legacy_scaling_grad_cpu - new_scaling_grad_torch).abs().max().template item<float>();
                float scaling_grad_mean_diff = (legacy_scaling_grad_cpu - new_scaling_grad_torch).abs().mean().template item<float>();

                spdlog::info("[{}] Scaling Gradient - Max diff: {:.2e}, Mean diff: {:.2e}",
                             iter, scaling_grad_max_diff, scaling_grad_mean_diff);

                // Log first 5 gradients (all 3 dimensions)
                spdlog::info("[{}] First 5 Scaling Gradients:", iter);
                for (int i = 0; i < std::min(5, static_cast<int>(legacy_scaling_grad_cpu.size(0))); ++i) {
                    auto legacy_row = legacy_scaling_grad_cpu[i];
                    auto new_row = new_scaling_grad_torch[i];
                    spdlog::info("[{}]   Gaussian {}: Legacy=[{:.6e}, {:.6e}, {:.6e}], New=[{:.6e}, {:.6e}, {:.6e}]",
                                 iter, i,
                                 legacy_row[0].template item<float>(), legacy_row[1].template item<float>(), legacy_row[2].template item<float>(),
                                 new_row[0].template item<float>(), new_row[1].template item<float>(), new_row[2].template item<float>());
                }
            }
        }

        // ============================================================
        // 5. OPTIMIZER STEP
        // ============================================================
        spdlog::info("[{}] === Optimizer Step ===", iter);

        // ============================================================
        // 5.5. COMPARE ADAM STATE AFTER STEP (First 2 iterations)
        // NOTE: We compare AFTER step to see the momentum values
        // ============================================================

        // Call strategy step for both pipelines
        // Note: post_backward needs to be called before step to populate visibility info
        legacy_init.strategy->post_backward(iter, legacy_output);
        new_init.strategy->post_backward(iter, new_output);

        // Perform optimizer step
        legacy_init.strategy->step(iter);
        new_init.strategy->step(iter);

        spdlog::info("[{}] Optimizer step complete", iter);

        // ============================================================
        // 5.6. COMPARE ADAM MOMENTUM STATE (First 2 iterations)
        // ============================================================
        if (iter <= 2) {
            spdlog::info("[{}] === Adam Momentum State Comparison (AFTER step) ===", iter);

            // Get optimizers
            auto* new_opt = dynamic_cast<lfs::training::MCMC*>(new_init.strategy.get())->get_optimizer();
            auto* legacy_opt = dynamic_cast<gs::training::MCMC*>(legacy_init.strategy.get())->get_optimizer();

            // Get legacy model to find which param group is opacity
            auto& legacy_model = legacy_init.strategy->get_model();
            auto legacy_opacity_raw = legacy_model.opacity_raw();

            // PyTorch optimizer state is stored per-parameter
            // The state is accessed via state() map keyed by parameter pointer
            // Adam stores: step, exp_avg, exp_avg_sq

            // Find the parameter group index for opacity
            // In MCMC, params are ordered: [means, sh0, shN, scaling, rotation, opacity]
            // So opacity should be param_group 5 (0-indexed)
            int opacity_param_idx = 5;

            spdlog::info("[{}] === OPACITY MOMENTUM COMPARISON ===", iter);

            // Get legacy state
            try {
                auto& param_groups = legacy_opt->param_groups();
                if (opacity_param_idx < param_groups.size()) {
                    auto& opacity_param_group = param_groups[opacity_param_idx];
                    auto& params = opacity_param_group.params();

                    if (!params.empty()) {
                        auto& opacity_param = params[0];

                        // Try to access state
                        auto& state = legacy_opt->state();
                        auto state_it = state.find(opacity_param.unsafeGetTensorImpl());

                        if (state_it != state.end()) {
                            // Cast state to FusedAdam::AdamParamState
                            auto* adam_state = static_cast<gs::training::FusedAdam::AdamParamState*>(state_it->second.get());

                            spdlog::info("[{}] Legacy Opacity State:", iter);
                            spdlog::info("[{}]   Step count: {}", iter, adam_state->step_count);

                            auto legacy_exp_avg = adam_state->exp_avg.cpu();
                            auto legacy_exp_avg_sq = adam_state->exp_avg_sq.cpu();

                            auto exp_avg_sizes = legacy_exp_avg.sizes();
                            auto exp_avg_sq_sizes = legacy_exp_avg_sq.sizes();
                            spdlog::info("[{}]   exp_avg shape: [{}]", iter, legacy_exp_avg.numel());
                            spdlog::info("[{}]   exp_avg_sq shape: [{}]", iter, legacy_exp_avg_sq.numel());

                            // Get new state
                            auto* new_opacity_state = new_opt->get_state(lfs::training::ParamType::Opacity);
                            if (new_opacity_state) {
                                spdlog::info("[{}] New Opacity State:", iter);
                                spdlog::info("[{}]   Step count: {}", iter, new_opacity_state->step_count);

                                auto new_exp_avg_cpu = new_opacity_state->exp_avg.to(lfs::core::Device::CPU);
                                auto new_exp_avg_sq_cpu = new_opacity_state->exp_avg_sq.to(lfs::core::Device::CPU);

                                // Compare first 5 values side by side
                                spdlog::info("[{}] First 5 Opacity exp_avg (Legacy vs New):", iter);
                                const float* legacy_avg_ptr = legacy_exp_avg.template data_ptr<float>();
                                const float* new_avg_ptr = new_exp_avg_cpu.ptr<float>();

                                for (int i = 0; i < std::min(5, static_cast<int>(legacy_exp_avg.numel())); ++i) {
                                    float legacy_val = legacy_avg_ptr[i];
                                    float new_val = new_avg_ptr[i];
                                    float diff = std::abs(legacy_val - new_val);
                                    spdlog::info("[{}]   [{}]: Legacy={:.6e}, New={:.6e}, Diff={:.6e}",
                                                 iter, i, legacy_val, new_val, diff);
                                }

                                spdlog::info("[{}] First 5 Opacity exp_avg_sq (Legacy vs New):", iter);
                                const float* legacy_sq_ptr = legacy_exp_avg_sq.template data_ptr<float>();
                                const float* new_sq_ptr = new_exp_avg_sq_cpu.ptr<float>();

                                for (int i = 0; i < std::min(5, static_cast<int>(legacy_exp_avg_sq.numel())); ++i) {
                                    float legacy_val = legacy_sq_ptr[i];
                                    float new_val = new_sq_ptr[i];
                                    float diff = std::abs(legacy_val - new_val);
                                    spdlog::info("[{}]   [{}]: Legacy={:.6e}, New={:.6e}, Diff={:.6e}",
                                                 iter, i, legacy_val, new_val, diff);
                                }

                                // Compute overall statistics
                                torch::Tensor new_exp_avg_torch = torch::from_blob(
                                    const_cast<float*>(new_avg_ptr),
                                    {static_cast<long>(new_opacity_state->exp_avg.numel())},
                                    torch::kFloat32
                                ).clone();

                                auto exp_avg_diff = (legacy_exp_avg - new_exp_avg_torch).abs();
                                spdlog::info("[{}] Opacity exp_avg: Max diff={:.6e}, Mean diff={:.6e}",
                                             iter, exp_avg_diff.max().item().toFloat(), exp_avg_diff.mean().item().toFloat());
                            }
                        } else {
                            spdlog::warn("[{}] Legacy opacity state not found in optimizer", iter);
                        }
                    }
                }
            } catch (const std::exception& e) {
                spdlog::error("[{}] Failed to access legacy optimizer state: {}", iter, e.what());
            }
        }

        // ============================================================
        // 5.5. CHECK SH0 VALUES AFTER OPTIMIZER STEP (iteration 1 only)
        // ============================================================
        if (iter == 1) {
            spdlog::info("[{}] === SH0 Memory Analysis AFTER Optimizer Step ===", iter);

            auto& legacy_model_post = legacy_init.strategy->get_model();
            auto& new_model_post = new_init.strategy->get_model();

            // Legacy SH0
            auto legacy_sh0_post = legacy_model_post.sh0();
            const float* legacy_sh0_ptr_post = legacy_sh0_post.template data_ptr<float>();

            // New SH0
            auto& new_sh0_post = new_model_post.sh0();
            const float* new_sh0_ptr_post = new_sh0_post.ptr<float>();

            // Check first 3 Gaussians using tensor indexing (safer)
            spdlog::info("[{}] First 3 Gaussians AFTER optimizer:", iter);
            auto legacy_sh0_cpu_post_check = legacy_sh0_post.cpu().reshape({legacy_sh0_post.size(0), 3});
            auto new_sh0_flat_post_check = new_sh0_post.contiguous().flatten(1, 2);
            auto new_sh0_cpu_post_check = new_sh0_flat_post_check.to(lfs::core::Device::CPU);
            size_t N_post_check = new_sh0_cpu_post_check.shape()[0];
            torch::Tensor new_sh0_torch_post_check = torch::empty({static_cast<long>(N_post_check), 3}, torch::kFloat32);
            std::memcpy(new_sh0_torch_post_check.template data_ptr<float>(), new_sh0_cpu_post_check.ptr<float>(), N_post_check * 3 * sizeof(float));

            for (int i = 0; i < 3; ++i) {
                auto legacy_row = legacy_sh0_cpu_post_check[i];
                auto new_row = new_sh0_torch_post_check[i];
                spdlog::info("[{}]   Gaussian {}: Legacy=[{:.6f}, {:.6f}, {:.6f}], New=[{:.6f}, {:.6f}, {:.6f}]",
                           iter, i,
                           legacy_row[0].item().toFloat(), legacy_row[1].item().toFloat(), legacy_row[2].item().toFloat(),
                           new_row[0].item().toFloat(), new_row[1].item().toFloat(), new_row[2].item().toFloat());

                // Compute per-Gaussian difference
                float diff_r = std::abs(legacy_row[0].item().toFloat() - new_row[0].item().toFloat());
                float diff_g = std::abs(legacy_row[1].item().toFloat() - new_row[1].item().toFloat());
                float diff_b = std::abs(legacy_row[2].item().toFloat() - new_row[2].item().toFloat());
                spdlog::info("[{}]     Diff: R={:.6f}, G={:.6f}, B={:.6f}", iter, diff_r, diff_g, diff_b);
            }

            // Compute overall difference
            auto legacy_sh0_cpu_post = legacy_sh0_post.cpu().reshape({legacy_sh0_post.size(0), 3});
            auto new_sh0_flat_post = new_sh0_post.contiguous().flatten(1, 2);
            auto new_sh0_cpu_post = new_sh0_flat_post.to(lfs::core::Device::CPU);
            size_t N_post = new_sh0_cpu_post.shape()[0];
            size_t C_post = new_sh0_cpu_post.shape()[1];
            torch::Tensor new_sh0_torch_post = torch::empty({static_cast<long>(N_post), static_cast<long>(C_post)}, torch::kFloat32);
            std::memcpy(new_sh0_torch_post.template data_ptr<float>(), new_sh0_cpu_post.ptr<float>(), N_post * C_post * sizeof(float));

            float sh0_diff_after = (legacy_sh0_cpu_post - new_sh0_torch_post).abs().max().item().toFloat();
            float sh0_mean_diff_after = (legacy_sh0_cpu_post - new_sh0_torch_post).abs().mean().item().toFloat();
            spdlog::info("[{}] SH0 AFTER optimizer: max diff={:.6f}, mean diff={:.6f}", iter, sh0_diff_after, sh0_mean_diff_after);
        }

        // ============================================================
        // 6. COMPARE OPTIMIZER STATE (betas, learning rates)
        // ============================================================
        spdlog::info("[{}] === Optimizer State Comparison ===", iter);

        // Get optimizer instances
        auto* legacy_opt = legacy_init.strategy->get_optimizer();
        auto* new_opt = new_init.strategy->get_optimizer();

        // Compare learning rates for each parameter group
        spdlog::info("[{}] Learning Rates:", iter);
        const char* param_names[] = {"Means", "Sh0", "ShN", "Scaling", "Rotation", "Opacity"};
        lfs::training::ParamType param_types[] = {
            lfs::training::ParamType::Means,
            lfs::training::ParamType::Sh0,
            lfs::training::ParamType::ShN,
            lfs::training::ParamType::Scaling,
            lfs::training::ParamType::Rotation,
            lfs::training::ParamType::Opacity
        };
        for (size_t i = 0; i < 6; ++i) {
            double legacy_lr = legacy_init.strategy->get_lr(i);
            float new_lr = new_opt->get_param_lr(param_types[i]);  // Get per-parameter LR

            spdlog::info("[{}]   {}: Legacy={:.2e}, New={:.2e}",
                         iter, param_names[i], legacy_lr, new_lr);
        }

        // Compare betas (Adam hyperparameters)
        auto legacy_adam_options = static_cast<torch::optim::AdamOptions&>(legacy_opt->param_groups()[0].options());
        spdlog::info("[{}] Beta1: Legacy={:.6f}, New={:.6f}",
                     iter,
                     std::get<0>(legacy_adam_options.betas()),
                     0.9f);  // New optimizer uses fixed betas (TODO: expose config)
        spdlog::info("[{}] Beta2: Legacy={:.6f}, New={:.6f}",
                     iter,
                     std::get<1>(legacy_adam_options.betas()),
                     0.999f);  // New optimizer uses fixed betas

        // ============================================================
        // 7. COMPARE ALL GAUSSIAN PARAMETERS (after optimizer step)
        // ============================================================
        spdlog::info("[{}] === Gaussian Parameter Comparison (all attributes) ===", iter);

        auto& legacy_model = legacy_init.strategy->get_model();
        auto& new_model = new_init.strategy->get_model();

        // Helper lambda to compare tensors
        auto compare_tensors = [&](const torch::Tensor& legacy_t, const lfs::core::Tensor& new_t, const char* name) {
            auto new_cpu = new_t.to(lfs::core::Device::CPU);

            // Build shape vector
            std::vector<long> shape_vec;
            for (size_t i = 0; i < new_cpu.ndim(); ++i) {
                shape_vec.push_back(static_cast<long>(new_cpu.shape()[i]));
            }

            torch::Tensor new_torch = torch::from_blob(
                new_cpu.ptr<float>(),
                torch::IntArrayRef(shape_vec),
                torch::kFloat32).clone();

            float max_diff = (legacy_t.cpu() - new_torch).abs().max().template item<float>();
            float mean_diff = (legacy_t.cpu() - new_torch).abs().mean().template item<float>();

            spdlog::info("[{}]   {} - Max diff: {:.2e}, Mean diff: {:.2e}",
                         iter, name, max_diff, mean_diff);
        };

        // Compare all attributes
        compare_tensors(legacy_model.means(), new_model.means(), "Means");

        // SH0: flatten from [N, 1, 3] to [N, 3]
        auto sh0_flat = new_model.sh0().contiguous().flatten(1, 2);
        spdlog::info("[{}] SH0 shape after flatten: [{}, {}]", iter, sh0_flat.shape()[0], sh0_flat.shape()[1]);
        // IMPORTANT: Also reshape legacy SH0 to [N, 3] before comparison
        auto legacy_sh0_reshaped = legacy_model.sh0().reshape({legacy_model.sh0().size(0), 3});
        compare_tensors(legacy_sh0_reshaped, sh0_flat, "SH0");

        // ShN: Manual flattening workaround for tensor lib issue
        // Original shape: [N, 15, 3], need to compare as [N, 45]
        spdlog::info("[{}] Starting ShN comparison (manual flatten)", iter);
        {
            auto shN_cpu = new_model.shN().to(lfs::core::Device::CPU);
            size_t N = shN_cpu.shape()[0];
            spdlog::info("[{}] ShN CPU copy complete, N={}", iter, N);

            // Check legacy shN shape
            auto legacy_shN = legacy_model.shN().cpu();
            if (legacy_shN.dim() == 3) {
                spdlog::info("[{}] Legacy ShN shape: [{}  {}, {}]",
                            iter, legacy_shN.size(0), legacy_shN.size(1), legacy_shN.size(2));
            } else {
                spdlog::info("[{}] Legacy ShN shape: [{}, {}]",
                            iter, legacy_shN.size(0), legacy_shN.size(1));
            }

            // Manually create flattened torch tensor with shape [N, 45]
            torch::Tensor shN_torch = torch::empty({static_cast<long>(N), 45}, torch::kFloat32);
            float* dst = shN_torch.template data_ptr<float>();
            const float* src = shN_cpu.ptr<float>();
            // Copy data in row-major order
            std::memcpy(dst, src, N * 45 * sizeof(float));
            spdlog::info("[{}] ShN memcpy complete, shN_torch shape: [{}, {}]",  iter, shN_torch.size(0), shN_torch.size(1));

            // Reshape legacy if needed
            if (legacy_shN.dim() == 3) {
                legacy_shN = legacy_shN.reshape({static_cast<long>(N), 45});
            }

            float max_diff = (legacy_shN - shN_torch).abs().max().template item<float>();
            float mean_diff = (legacy_shN - shN_torch).abs().mean().template item<float>();
            spdlog::info("[{}]   ShN - Max diff: {:.2e}, Mean diff: {:.2e}",
                         iter, max_diff, mean_diff);
        }

        compare_tensors(legacy_model.scaling_raw(), new_model.scaling_raw(), "Scaling");
        compare_tensors(legacy_model.rotation_raw(), new_model.rotation_raw(), "Rotation");
        compare_tensors(legacy_model.opacity_raw(), new_model.opacity_raw(), "Opacity");

        // ============================================================
        // 8. COMPARE SH0 GRADIENTS (detailed analysis)
        // ============================================================
        spdlog::info("[{}] === SH0 Gradient Detailed Comparison ===", iter);
        auto legacy_sh0_grad = legacy_model.sh0().grad();
        auto& new_sh0_grad = new_model.sh0_grad();

        if (legacy_sh0_grad.defined() && new_sh0_grad.ndim() > 0) {
            // Legacy SH0 grad is [N, 1, 3], new is [N, 1, 3]
            auto legacy_sh0_grad_cpu = legacy_sh0_grad.cpu();
            auto new_sh0_grad_flat = new_sh0_grad.contiguous().flatten(1, 2);
            auto new_sh0_grad_cpu = new_sh0_grad_flat.to(lfs::core::Device::CPU);

            // Manual memcpy to convert to torch
            size_t N = new_sh0_grad_cpu.shape()[0];
            size_t C = new_sh0_grad_cpu.shape()[1];
            torch::Tensor new_sh0_grad_torch = torch::empty({static_cast<long>(N), static_cast<long>(C)}, torch::kFloat32);
            const float* src_ptr = new_sh0_grad_cpu.ptr<float>();
            float* dst_ptr = new_sh0_grad_torch.template data_ptr<float>();
            std::memcpy(dst_ptr, src_ptr, N * C * sizeof(float));

            // Reshape legacy if needed
            auto legacy_sh0_grad_reshaped = legacy_sh0_grad_cpu.reshape({legacy_sh0_grad_cpu.size(0), 3});

            float sh0_grad_max_diff = (legacy_sh0_grad_reshaped - new_sh0_grad_torch).abs().max().item().toFloat();
            float sh0_grad_mean_diff = (legacy_sh0_grad_reshaped - new_sh0_grad_torch).abs().mean().item().toFloat();

            spdlog::info("[{}] SH0 Gradient - Max diff: {:.2e}, Mean diff: {:.2e}",
                         iter, sh0_grad_max_diff, sh0_grad_mean_diff);

            // Check a few sample values
            spdlog::info("[{}] Sample SH0 gradients (first 3 points):", iter);
            for (int i = 0; i < std::min(3, static_cast<int>(N)); ++i) {
                auto legacy_row = legacy_sh0_grad_reshaped[i];
                auto new_row = new_sh0_grad_torch[i];
                spdlog::info("[{}]   Point {}: Legacy=[{:.6f}, {:.6f}, {:.6f}], New=[{:.6f}, {:.6f}, {:.6f}]",
                            iter, i,
                            legacy_row[0].item().toFloat(), legacy_row[1].item().toFloat(), legacy_row[2].item().toFloat(),
                            new_row[0].item().toFloat(), new_row[1].item().toFloat(), new_row[2].item().toFloat());
            }
        } else {
            spdlog::warn("[{}] SH0 gradient not available for comparison", iter);
        }

        spdlog::info("[{}] === Iteration Complete ===", iter);
    }

    spdlog::info("=== Training loop comparison complete ===");
    return {};
}

} // namespace gs::training_debug
