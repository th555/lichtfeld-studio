/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// TODO: Port these components to LibTorch-free implementation
// #include "components/bilateral_grid.hpp"        // Temporarily disabled - requires LibTorch
// #include "components/sparsity_optimizer.hpp"     // Temporarily disabled - requires LibTorch
// #include "components/poseopt.hpp"
// #include "core/events.hpp"                  // Temporarily disabled - requires LibTorch (gs_core)
#include "core_new/parameters.hpp"
#include "dataset.hpp"
// #include "lfs/kernels/bilateral_grid.cuh"   // Temporarily disabled - not using bilateral grid
#include "metrics/metrics.hpp"
#include "optimizer/scheduler.hpp"
#include "progress.hpp"
#include "project_new/project.hpp"  // Using old project system for now
// TODO: Port 3DGUT rasterizer to LibTorch-free implementation
// #include "rasterization/rasterizer.hpp"
#include "strategies/istrategy.hpp"
#include "core_new/camera.hpp"
#include "core_new/tensor.hpp"
#include <atomic>
#include <expected>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stop_token>

namespace lfs::training {
    class Trainer {
    public:
        // Constructor that takes ownership of strategy and shares datasets
        Trainer(std::shared_ptr<CameraDataset> dataset,
                std::unique_ptr<IStrategy> strategy,
                std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits);

        // Delete copy operations
        Trainer(const Trainer&) = delete;

        Trainer& operator=(const Trainer&) = delete;

        // Allow move operations
        Trainer(Trainer&&) = default;

        Trainer& operator=(Trainer&&) = default;

        ~Trainer();

        // Initialize trainer - must be called before training
        std::expected<void, std::string> initialize(const lfs::core::param::TrainingParameters& params);

        // Check if trainer is initialized
        bool isInitialized() const { return initialized_.load(); }

        // Main training method with stop token support
        std::expected<void, std::string> train(std::stop_token stop_token = {});

        // Control methods for GUI interaction
        void request_pause() { pause_requested_ = true; }
        void request_resume() { pause_requested_ = false; }
        void request_save() { save_requested_ = true; }
        void request_stop() { stop_requested_ = true; }

        bool is_paused() const { return is_paused_.load(); }
        bool is_running() const { return is_running_.load(); }
        bool is_training_complete() const { return training_complete_.load(); }
        bool has_stopped() const { return stop_requested_.load(); }

        // Get current training state
        int get_current_iteration() const { return current_iteration_.load(); }
        float get_current_loss() const { return current_loss_.load(); }

        // just for viewer to get model
        const IStrategy& get_strategy() const { return *strategy_; }

        // Allow viewer to lock for rendering
        std::shared_mutex& getRenderMutex() const { return render_mutex_; }

        const lfs::core::param::TrainingParameters& getParams() const { return params_; }

        std::shared_ptr<const lfs::core::Camera> getCamById(int camId) const;

        std::vector<std::shared_ptr<const lfs::core::Camera>> getCamList() const;

        void setProject(std::shared_ptr<lfs::project::Project> project) { lf_project_ = project; }

        void load_cameras_info();

    private:
        // Helper for deferred event emission to prevent deadlocks
        struct DeferredEvents {
            std::vector<std::function<void()>> events;

            template <typename Event>
            void add(Event&& e) {
                events.push_back([e = std::move(e)]() { e.emit(); });
            }

            ~DeferredEvents() {
                for (auto& e : events)
                    e();
            }
        };

        // Training step result
        enum class StepResult {
            Continue,
            Stop,
            Error
        };

        // Returns the background color to use at a given iteration
        lfs::core::Tensor& background_for_step(int iter);

        // Protected method for processing a single training step
        std::expected<StepResult, std::string> train_step(
            int iter,
            lfs::core::Camera* cam,
            lfs::core::Tensor gt_image,
            RenderMode render_mode,
            std::stop_token stop_token = {});

        // Compute photometric loss AND gradient manually (no autograd)
        // Returns GPU tensor for loss (avoid sync!)
        std::expected<std::pair<lfs::core::Tensor, lfs::core::Tensor>, std::string> compute_photometric_loss_with_gradient(
            const lfs::core::Tensor& rendered,
            const lfs::core::Tensor& gt_image,
            const lfs::core::param::OptimizationParameters& opt_params);

        // Returns GPU tensor for loss (avoid sync!)
        std::expected<lfs::core::Tensor, std::string> compute_scale_reg_loss(
            lfs::core::SplatData& splatData,
            const lfs::core::param::OptimizationParameters& opt_params);

        // Returns GPU tensor for loss (avoid sync!)
        std::expected<lfs::core::Tensor, std::string> compute_opacity_reg_loss(
            lfs::core::SplatData& splatData,
            const lfs::core::param::OptimizationParameters& opt_params);

        // Temporarily disabled - requires LibTorch
        // std::expected<std::pair<float, BilateralGridTVContext>, std::string> compute_bilateral_grid_tv_loss(
        //     const std::unique_ptr<BilateralGrid>& bilateral_grid,
        //     const lfs::core::param::OptimizationParameters& opt_params);

        // Sparsity-related methods (LibTorch-free) - Temporarily disabled
        // std::expected<std::pair<float, SparsityLossContext>, std::string> compute_sparsity_loss_forward(
        //     int iter,
        //     const lfs::core::SplatData& splatData);

        // std::expected<void, std::string> handle_sparsity_update(
        //     int iter,
        //     lfs::core::SplatData& splatData);

        // std::expected<void, std::string> apply_sparsity_pruning(
        //     int iter,
        //     lfs::core::SplatData& splatData);

        // Cleanup method for re-initialization
        void cleanup();

        // Temporarily disabled - requires LibTorch
        // std::expected<void, std::string> initialize_bilateral_grid();

        // Handle control requests
        void handle_control_requests(int iter, std::stop_token stop_token = {});

        void save_ply(const std::filesystem::path& save_path, int iter_num, bool join_threads = true);

        // Member variables
        std::shared_ptr<CameraDataset> base_dataset_;
        std::shared_ptr<CameraDataset> train_dataset_;
        std::shared_ptr<CameraDataset> val_dataset_;
        std::unique_ptr<IStrategy> strategy_;
        lfs::core::param::TrainingParameters params_;
        std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits_;

        lfs::core::Tensor background_{};
        lfs::core::Tensor bg_mix_buffer_;
        std::unique_ptr<TrainingProgress> progress_;
        size_t train_dataset_size_ = 0;

        // Bilateral grid components - Temporarily disabled (requires LibTorch)
        // std::unique_ptr<BilateralGrid> bilateral_grid_;
        // std::unique_ptr<lfs::training::AdamOptimizer> bilateral_grid_optimizer_;  // Use ported Adam
        // std::unique_ptr<WarmupExponentialLR> bilateral_grid_scheduler_;

        // TODO: Port pose optimization to LibTorch-free implementation
        // std::unique_ptr<PoseOptimizationModule> poseopt_module_; // Pose optimization module
        // std::unique_ptr<torch::optim::Adam> poseopt_optimizer_;  // Optimizer for pose optimization

        // Sparsity optimizer (LibTorch-free) - Temporarily disabled (requires LibTorch)
        // std::unique_ptr<ISparsityOptimizer> sparsity_optimizer_;

        // Metrics evaluator - handles all evaluation logic
        std::unique_ptr<lfs::training::MetricsEvaluator> evaluator_;

        // Single mutex that protects the model during training
        mutable std::shared_mutex render_mutex_;

        // Mutex for initialization to ensure thread safety
        mutable std::mutex init_mutex_;

        // Control flags for thread communication
        std::atomic<bool> pause_requested_{false};
        std::atomic<bool> save_requested_{false};
        std::atomic<bool> stop_requested_{false};
        std::atomic<bool> is_paused_{false};
        std::atomic<bool> is_running_{false};
        std::atomic<bool> training_complete_{false};
        std::atomic<bool> ready_to_start_{false};
        std::atomic<bool> initialized_{false};

        // Current training state
        std::atomic<int> current_iteration_{0};
        std::atomic<float> current_loss_{0.0f};

        // Callback system for async operations (LibTorch-free)
        std::function<void()> callback_;
        std::atomic<bool> callback_busy_{false};
        cudaStream_t callback_stream_ = nullptr;
        cudaEvent_t callback_launch_event_ = nullptr;

        // camera id to cam
        std::map<int, std::shared_ptr<const lfs::core::Camera>> m_cam_id_to_cam;

        // LichtFeld project
        std::shared_ptr<lfs::project::Project> lf_project_ = nullptr;
    };
} // namespace lfs::training
