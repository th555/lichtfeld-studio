/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/parameters.hpp"
#include "core_new/splat_data.hpp"
#include "optimizer/render_output.hpp"

namespace lfs::training {

    class IStrategy {
    public:
        virtual ~IStrategy() = default;

        virtual void initialize(const lfs::core::param::OptimizationParameters& optimParams) = 0;

        virtual void post_backward(int iter, RenderOutput& render_output) = 0;

        virtual void step(int iter) = 0;

        virtual bool is_refining(int iter) const = 0;

        // Get the underlying Gaussian model for rendering
        virtual lfs::core::SplatData& get_model() = 0;

        virtual const lfs::core::SplatData& get_model() const = 0;

        // Remove Gaussians based on mask
        virtual void remove_gaussians(const lfs::core::Tensor& mask) = 0;
    };
} // namespace lfs::training
