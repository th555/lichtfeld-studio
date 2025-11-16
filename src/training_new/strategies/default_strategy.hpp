/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "istrategy.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include <memory>

namespace lfs::training {
    // Forward declarations
    struct RenderOutput;

    class DefaultStrategy : public IStrategy {
    public:
        DefaultStrategy() = delete;

        DefaultStrategy(lfs::core::SplatData&& splat_data);

        DefaultStrategy(const DefaultStrategy&) = delete;

        DefaultStrategy& operator=(const DefaultStrategy&) = delete;

        DefaultStrategy(DefaultStrategy&&) = default;

        DefaultStrategy& operator=(DefaultStrategy&&) = default;

        // IStrategy interface implementation
        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;

        void post_backward(int iter, RenderOutput& render_output) override;

        void step(int iter) override;

        bool is_refining(int iter) const override;

        lfs::core::SplatData& get_model() override { return _splat_data; }
        const lfs::core::SplatData& get_model() const override { return _splat_data; }

        void remove_gaussians(const lfs::core::Tensor& mask) override;

    private:
        // Helper functions
        void duplicate(const lfs::core::Tensor& is_duplicated);

        void split(const lfs::core::Tensor& is_split);

        void grow_gs(int iter);

        void remove(const lfs::core::Tensor& is_prune);

        void prune_gs(int iter);

        void reset_opacity();

        // Member variables
        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData _splat_data;
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;
    };
} // namespace lfs::training
