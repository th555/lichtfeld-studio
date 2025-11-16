/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "istrategy.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include <memory>

namespace lfs::training {

    class MCMC : public IStrategy {
    public:
        MCMC() = delete;
        explicit MCMC(lfs::core::SplatData&& splat_data);

        MCMC(const MCMC&) = delete;
        MCMC& operator=(const MCMC&) = delete;

        MCMC(MCMC&&) = default;
        MCMC& operator=(MCMC&&) = default;

        // IStrategy interface implementation
        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;
        void post_backward(int iter, RenderOutput& render_output) override;
        bool is_refining(int iter) const override;
        void step(int iter) override;

        lfs::core::SplatData& get_model() override { return _splat_data; }
        const lfs::core::SplatData& get_model() const override { return _splat_data; }

        void remove_gaussians(const lfs::core::Tensor& mask) override;

        // Accessor for debugging/comparison
        AdamOptimizer* get_optimizer() { return _optimizer.get(); }

        // Exposed for testing (compare with legacy implementation)
        int add_new_gs_test() { return add_new_gs(); }
        int add_new_gs_with_indices_test(const lfs::core::Tensor& sampled_idxs);
        int relocate_gs_test() { return relocate_gs(); }

    private:
        // Helper functions
        lfs::core::Tensor multinomial_sample(const lfs::core::Tensor& weights, int n, bool replacement = true);
        int relocate_gs();
        int add_new_gs();
        void inject_noise();
        void update_optimizer_for_relocate(const lfs::core::Tensor& sampled_indices,
                                          const lfs::core::Tensor& dead_indices,
                                          ParamType param_type);

        // Member variables
        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData _splat_data;
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;

        // MCMC specific parameters
        const float _noise_lr = 5e5f;

        // State variables
        lfs::core::Tensor _binoms;  // [n_max, n_max] binomial coefficients
    };

} // namespace lfs::training
