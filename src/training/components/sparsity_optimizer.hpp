/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include <expected>
#include <memory>
#include <string>
#include <torch/torch.h>

namespace gs::training {

    /**
     * @brief Interface for sparsity optimization methods
     *
     * Provides a clean abstraction for different sparsity-inducing techniques
     * that can be applied during Gaussian Splatting training.
     */
    class ISparsityOptimizer {
    public:
        virtual ~ISparsityOptimizer() = default;

        /**
         * @brief Initialize the optimizer with initial opacities
         * @param opacities Initial opacity values from the model
         * @return Error string if initialization fails
         */
        virtual std::expected<void, std::string> initialize(const torch::Tensor& opacities) = 0;

        /**
         * @brief Compute the sparsity regularization loss
         * @param opacities Current opacity values from the model
         * @return Loss tensor or error string
         */
        virtual std::expected<torch::Tensor, std::string> compute_loss(const torch::Tensor& opacities) const = 0;

        /**
         * @brief Update internal state (called periodically during training)
         * @param opacities Current opacity values from the model
         * @return Error string if update fails
         */
        virtual std::expected<void, std::string> update_state(const torch::Tensor& opacities) = 0;

        /**
         * @brief Get mask indicating which Gaussians to prune
         * @param opacities Current opacity values from the model
         * @return Boolean mask tensor or error string
         */
        virtual std::expected<torch::Tensor, std::string> get_prune_mask(const torch::Tensor& opacities) const = 0;

        /**
         * @brief Check if we should update state at this iteration
         */
        virtual bool should_update(int iter) const = 0;

        /**
         * @brief Check if we should apply loss at this iteration
         */
        virtual bool should_apply_loss(int iter) const = 0;

        /**
         * @brief Check if we should prune at this iteration
         */
        virtual bool should_prune(int iter) const = 0;

        /**
         * @brief Get the number of Gaussians that will be pruned
         */
        virtual int get_num_to_prune(const torch::Tensor& opacities) const = 0;

        /**
         * @brief Check if the optimizer has been initialized
         */
        virtual bool is_initialized() const = 0;
    };

    /**
     * @brief ADMM-based sparsity optimizer
     *
     * Implements Alternating Direction Method of Multipliers (ADMM) for
     * inducing sparsity in Gaussian opacity values during training.
     */
    class ADMMSparsityOptimizer : public ISparsityOptimizer {
    public:
        struct Config {
            int sparsify_steps = 15000;  // Total steps for sparsification
            float init_rho = 0.0005f;    // ADMM penalty parameter
            float prune_ratio = 0.6f;    // Final pruning ratio
            int update_every = 50;       // Update ADMM state every N iterations
            int start_iteration = 30000; // When to start sparsification (after base training)
        };

        explicit ADMMSparsityOptimizer(const Config& config);

        std::expected<void, std::string> initialize(const torch::Tensor& opacities) override;
        std::expected<torch::Tensor, std::string> compute_loss(const torch::Tensor& opacities) const override;
        std::expected<void, std::string> update_state(const torch::Tensor& opacities) override;
        std::expected<torch::Tensor, std::string> get_prune_mask(const torch::Tensor& opacities) const override;

        bool should_update(int iter) const override {
            int relative_iter = iter - config_.start_iteration;
            return iter >= config_.start_iteration &&
                   relative_iter > 0 &&
                   relative_iter < config_.sparsify_steps &&
                   relative_iter % config_.update_every == 0;
        }

        bool should_apply_loss(int iter) const override {
            return iter >= config_.start_iteration &&
                   iter < (config_.start_iteration + config_.sparsify_steps);
        }

        bool should_prune(int iter) const override {
            return iter == (config_.start_iteration + config_.sparsify_steps);
        }

        int get_num_to_prune(const torch::Tensor& opacities) const override;

        bool is_initialized() const override { return initialized_; }

        // Test/comparison helpers - wrap around compute_loss() to match new API signature
        // Context type for forward pass (includes internal state for debugging)
        struct ComputeLossContext {
            torch::Tensor opacities_with_grad;
            torch::Tensor opa_sigmoid;  // sigmoid(opacities) for debugging
            torch::Tensor z;  // z variable for debugging
            torch::Tensor u;  // u variable for debugging
        };

        std::expected<std::pair<torch::Tensor, ComputeLossContext>, std::string>
        compute_loss_forward(const torch::Tensor& opacities) const {
            auto opacities_with_grad = opacities.requires_grad_(true);
            auto loss_result = compute_loss(opacities_with_grad);
            if (!loss_result.has_value()) {
                return std::unexpected(loss_result.error());
            }
            // Populate context with internal state for debugging
            auto opa_sigmoid = torch::sigmoid(opacities_with_grad);
            ComputeLossContext ctx{opacities_with_grad, opa_sigmoid, z_, u_};
            return std::make_pair(loss_result.value(), ctx);
        }

        std::expected<torch::Tensor, std::string>
        compute_loss_backward(const ComputeLossContext& ctx, float grad_loss) const {
            // Recompute loss to build computation graph
            auto loss_result = compute_loss(ctx.opacities_with_grad);
            if (!loss_result.has_value()) {
                return std::unexpected(loss_result.error());
            }

            // Clear any existing gradients
            if (ctx.opacities_with_grad.grad().defined()) {
                ctx.opacities_with_grad.grad().zero_();
            }

            // Backward pass
            auto loss = loss_result.value();
            loss.backward(torch::tensor(grad_loss, loss.options()));

            return ctx.opacities_with_grad.grad();
        }

    private:
        /**
         * @brief Apply soft thresholding to enforce sparsity
         * @param z Input tensor
         * @return Thresholded tensor
         */
        torch::Tensor prune_z(const torch::Tensor& z) const;

        Config config_;
        torch::Tensor u_; // Dual variable (Lagrange multiplier)
        torch::Tensor z_; // Auxiliary variable for sparsity
        bool initialized_ = false;
    };

    /**
     * @brief Factory for creating sparsity optimizers
     */
    class SparsityOptimizerFactory {
    public:
        static std::unique_ptr<ISparsityOptimizer> create(
            const std::string& method,
            const ADMMSparsityOptimizer::Config& config);
    };

} // namespace gs::training