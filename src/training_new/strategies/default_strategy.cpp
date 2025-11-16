/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "default_strategy.hpp"
#include "core_new/logger.hpp"
#include "core_new/parameters.hpp"
#include "optimizer/render_output.hpp"
#include "strategy_utils.hpp"
#include "kernels/densification_kernels.hpp"

namespace lfs::training {
    DefaultStrategy::DefaultStrategy(lfs::core::SplatData&& splat_data)
        : _splat_data(std::move(splat_data)) {
    }

    void DefaultStrategy::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        initialize_gaussians(_splat_data);

        // Initialize optimizer
        _optimizer = create_optimizer(_splat_data, *_params);

        // Initialize exponential scheduler
        _scheduler = create_scheduler(*_params, *_optimizer);

        // Initialize densification info: [2, N] tensor for tracking gradients
        _splat_data._densification_info = lfs::core::Tensor::zeros(
            {2, static_cast<size_t>(_splat_data.size())},
            _splat_data.means().device());
    }

    bool DefaultStrategy::is_refining(int iter) const {
        return (iter > _params->start_refine &&
                iter % _params->refine_every == 0 &&
                iter % _params->reset_every >= _params->pause_refine_after_reset);
    }

    void DefaultStrategy::remove_gaussians(const lfs::core::Tensor& mask) {
        int mask_sum = mask.to(lfs::core::DataType::Int32).sum().template item<int>();

        if (mask_sum == 0) {
            LOG_DEBUG("No Gaussians to remove");
            return;
        }

        LOG_DEBUG("Removing {} Gaussians", mask_sum);
        remove(mask);
    }

    void DefaultStrategy::duplicate(const lfs::core::Tensor& is_duplicated) {
        const lfs::core::Tensor sampled_idxs = is_duplicated.nonzero().squeeze(-1);
        const int num_duplicated = sampled_idxs.shape()[0];

        if (num_duplicated == 0) {
            return;  // Nothing to duplicate
        }

        // Gather parameters for selected Gaussians to duplicate
        LOG_DEBUG("duplicate(): sh0 shape before index_select: {}", _splat_data.sh0().shape().str());
        LOG_DEBUG("duplicate(): shN shape before index_select: {}", _splat_data.shN().shape().str());

        auto pos_selected = _splat_data.means().index_select(0, sampled_idxs).contiguous();
        auto rot_selected = _splat_data.rotation_raw().index_select(0, sampled_idxs).contiguous();
        auto scale_selected = _splat_data.scaling_raw().index_select(0, sampled_idxs).contiguous();
        auto sh0_selected = _splat_data.sh0().index_select(0, sampled_idxs).contiguous();
        auto shN_selected = _splat_data.shN().index_select(0, sampled_idxs).contiguous();
        auto op_selected = _splat_data.opacity_raw().index_select(0, sampled_idxs).contiguous();

        LOG_DEBUG("duplicate(): sh0_selected shape after index_select: {}", sh0_selected.shape().str());
        LOG_DEBUG("duplicate(): shN_selected shape after index_select: {}", shN_selected.shape().str());

        // Concatenate parameters directly in SplatData
        LOG_DEBUG("duplicate(): About to cat means: existing={}, selected={}", _splat_data.means().shape().str(), pos_selected.shape().str());
        _splat_data.means() = _splat_data.means().cat(pos_selected, 0);

        LOG_DEBUG("duplicate(): About to cat rotation: existing={}, selected={}", _splat_data.rotation_raw().shape().str(), rot_selected.shape().str());
        _splat_data.rotation_raw() = _splat_data.rotation_raw().cat(rot_selected, 0);

        LOG_DEBUG("duplicate(): About to cat scaling: existing={}, selected={}", _splat_data.scaling_raw().shape().str(), scale_selected.shape().str());
        _splat_data.scaling_raw() = _splat_data.scaling_raw().cat(scale_selected, 0);

        LOG_ERROR("duplicate(): About to cat sh0: existing={}, selected={}", _splat_data.sh0().shape().str(), sh0_selected.shape().str());
        _splat_data.sh0() = _splat_data.sh0().cat(sh0_selected, 0);

        LOG_ERROR("duplicate(): About to cat shN: existing={}, selected={}", _splat_data.shN().shape().str(), shN_selected.shape().str());
        _splat_data.shN() = _splat_data.shN().cat(shN_selected, 0);

        LOG_DEBUG("duplicate(): About to cat opacity: existing={}, selected={}", _splat_data.opacity_raw().shape().str(), op_selected.shape().str());
        _splat_data.opacity_raw() = _splat_data.opacity_raw().cat(op_selected, 0);

        // Update gradients to match new size
        if (_splat_data.has_gradients()) {
            auto means_shape = _splat_data.means().shape();
            auto rotation_shape = _splat_data.rotation_raw().shape();
            auto scaling_shape = _splat_data.scaling_raw().shape();
            auto sh0_shape = _splat_data.sh0().shape();
            auto shN_shape = _splat_data.shN().shape();
            auto opacity_shape = _splat_data.opacity_raw().shape();

            _splat_data.means_grad() = lfs::core::Tensor::zeros(means_shape, _splat_data.means().device());
            _splat_data.rotation_grad() = lfs::core::Tensor::zeros(rotation_shape, _splat_data.rotation_raw().device());
            _splat_data.scaling_grad() = lfs::core::Tensor::zeros(scaling_shape, _splat_data.scaling_raw().device());
            _splat_data.sh0_grad() = lfs::core::Tensor::zeros(sh0_shape, _splat_data.sh0().device());
            _splat_data.shN_grad() = lfs::core::Tensor::zeros(shN_shape, _splat_data.shN().device());
            _splat_data.opacity_grad() = lfs::core::Tensor::zeros(opacity_shape, _splat_data.opacity_raw().device());
        }

        // Update optimizer states: duplicate the optimizer states for the duplicated Gaussians
        // Following the same pattern as split()
        auto update_optimizer_state = [&](ParamType param_type, const char* name) {
            if (const auto* state = _optimizer->get_state(param_type)) {
                LOG_DEBUG("update_optimizer_state for {}: state->exp_avg shape = {}", name, state->exp_avg.shape().str());

                // Keep all existing states
                lfs::core::Tensor existing_exp_avg = state->exp_avg;
                lfs::core::Tensor existing_exp_avg_sq = state->exp_avg_sq;

                // Get the states for the gaussians we're duplicating
                lfs::core::Tensor dup_exp_avg = state->exp_avg.index_select(0, sampled_idxs);
                lfs::core::Tensor dup_exp_avg_sq = state->exp_avg_sq.index_select(0, sampled_idxs);

                LOG_DEBUG("  existing_exp_avg shape: {}", existing_exp_avg.shape().str());
                LOG_DEBUG("  dup_exp_avg shape: {}", dup_exp_avg.shape().str());

                // Create new state and set it
                AdamParamState new_state = *state;
                new_state.exp_avg = existing_exp_avg.cat(dup_exp_avg, 0);
                new_state.exp_avg_sq = existing_exp_avg_sq.cat(dup_exp_avg_sq, 0);
                // Update size to match new parameter count
                size_t old_size = state->size;
                size_t new_size = state->size + num_duplicated;
                new_state.size = new_size;
                LOG_DEBUG("  duplicate() size update: {} -> {} (added {})", old_size, new_size, num_duplicated);
                _optimizer->set_state(param_type, new_state);
            }
        };

        update_optimizer_state(ParamType::Means, "means");
        update_optimizer_state(ParamType::Rotation, "rotation");
        update_optimizer_state(ParamType::Scaling, "scaling");
        update_optimizer_state(ParamType::Sh0, "sh0");
        update_optimizer_state(ParamType::ShN, "shN");
        update_optimizer_state(ParamType::Opacity, "opacity");
    }

    void DefaultStrategy::split(const lfs::core::Tensor& is_split) {
        const lfs::core::Tensor split_idxs = is_split.nonzero().squeeze(-1);
        const lfs::core::Tensor keep_idxs = is_split.logical_not().nonzero().squeeze(-1);

        const int N = _splat_data.size();
        const int num_split = split_idxs.shape()[0];
        const int num_keep = keep_idxs.shape()[0];

        if (num_split == 0) {
            return;  // Nothing to split
        }

        constexpr int split_size = 2;

        // Get SH dimensions - total elements per Gaussian (coeffs * channels)
        // shN has shape [N, num_coeffs, 3], so total elements = num_coeffs * 3
        const int shN_coeffs = _splat_data.shN().shape()[1];
        const int shN_channels = _splat_data.shN().shape()[2];
        const int shN_dim = shN_coeffs * shN_channels;  // Total elements per Gaussian

        // Generate random noise [2, num_split, 3]
        const lfs::core::Tensor random_noise = lfs::core::Tensor::randn(
            {split_size, static_cast<size_t>(num_split), 3},
            _splat_data.sh0().device());

        // Allocate output tensors [num_keep + num_split*2, ...]
        const int out_size = num_keep + num_split * split_size;
        auto positions_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 3}, _splat_data.means().device());
        auto rotations_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 4}, _splat_data.rotation_raw().device());
        auto scales_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 3}, _splat_data.scaling_raw().device());
        // CUDA kernel produces 2D outputs, we'll reshape to 3D after
        auto sh0_out_2d = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 3}, _splat_data.sh0().device());
        auto shN_out_2d = lfs::core::Tensor::empty({static_cast<size_t>(out_size), static_cast<size_t>(shN_dim)}, _splat_data.shN().device());
        auto opacities_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 1}, _splat_data.opacity_raw().device());

        // Call custom CUDA kernel (outputs sh0 and shN separately - NO slice/contiguous overhead!)
        kernels::launch_split_gaussians(
            _splat_data.means().ptr<float>(),
            _splat_data.rotation_raw().ptr<float>(),
            _splat_data.scaling_raw().ptr<float>(),
            _splat_data.sh0().ptr<float>(),
            _splat_data.shN().ptr<float>(),
            _splat_data.opacity_raw().ptr<float>(),
            positions_out.ptr<float>(),
            rotations_out.ptr<float>(),
            scales_out.ptr<float>(),
            sh0_out_2d.ptr<float>(),
            shN_out_2d.ptr<float>(),
            opacities_out.ptr<float>(),
            split_idxs.ptr<int64_t>(),
            keep_idxs.ptr<int64_t>(),
            random_noise.ptr<float>(),
            N,
            num_split,
            num_keep,
            shN_dim,
            _params->revised_opacity,
            nullptr  // default stream
        );

        // Reshape sh0 and shN from flat 1D (per Gaussian) to original 3D structure
        // sh0: [N, 3] -> [N, 1, 3] (3 elements = 1 coeff * 3 channels)
        // shN: [N, 45] -> [N, 15, 3] (45 elements = 15 coeffs * 3 channels)
        auto sh0_out = sh0_out_2d.reshape({out_size, 1, 3});
        auto shN_out = shN_out_2d.reshape({out_size, shN_coeffs, shN_channels});

        // Update SplatData with new tensors (already contiguous from kernel!)
        _splat_data.means() = positions_out;
        _splat_data.rotation_raw() = rotations_out;
        _splat_data.scaling_raw() = scales_out;
        _splat_data.sh0() = sh0_out;
        _splat_data.shN() = shN_out;
        _splat_data.opacity_raw() = opacities_out.squeeze(-1);

        // Update gradients to match new size
        if (_splat_data.has_gradients()) {
            _splat_data.means_grad() = lfs::core::Tensor::zeros(positions_out.shape(), positions_out.device());
            _splat_data.rotation_grad() = lfs::core::Tensor::zeros(rotations_out.shape(), rotations_out.device());
            _splat_data.scaling_grad() = lfs::core::Tensor::zeros(scales_out.shape(), scales_out.device());
            _splat_data.sh0_grad() = lfs::core::Tensor::zeros(sh0_out.shape(), sh0_out.device());
            _splat_data.shN_grad() = lfs::core::Tensor::zeros(shN_out.shape(), shN_out.device());
            _splat_data.opacity_grad() = lfs::core::Tensor::zeros(opacities_out.squeeze(-1).shape(), opacities_out.device());
        }

        // Update optimizer states: keep old states for kept Gaussians, add zeros for split Gaussians
        auto update_optimizer_state = [&](ParamType param_type) {
            if (const auto* state = _optimizer->get_state(param_type)) {
                LOG_DEBUG("split() updating optimizer state for param_type={}", static_cast<int>(param_type));
                LOG_DEBUG("  Old state.size = {}", state->size);
                LOG_DEBUG("  num_keep = {}, num_split = {}, split_size = {}", num_keep, num_split, split_size);

                // Keep states for kept Gaussians
                lfs::core::Tensor keep_exp_avg = state->exp_avg.index_select(0, keep_idxs);
                lfs::core::Tensor keep_exp_avg_sq = state->exp_avg_sq.index_select(0, keep_idxs);

                // CRITICAL: Create zeros tensor with same shape as existing state (preserving all dimensions!)
                // Get the shape from existing state, but update the first dimension for new Gaussians
                auto existing_shape = state->exp_avg.shape();
                std::vector<size_t> zero_dims(existing_shape.rank());
                zero_dims[0] = num_split * split_size;  // New Gaussians from split
                for (size_t i = 1; i < existing_shape.rank(); i++) {
                    zero_dims[i] = existing_shape[i];  // Keep all other dimensions
                }

                auto zeros = lfs::core::Tensor::zeros(
                    lfs::core::TensorShape(zero_dims),
                    state->exp_avg.device(),
                    state->exp_avg.dtype());

                LOG_DEBUG("  keep_exp_avg shape: {}, zeros shape: {}", keep_exp_avg.shape().str(), zeros.shape().str());

                // Create new state and set it
                AdamParamState new_state = *state;
                new_state.exp_avg = keep_exp_avg.cat(zeros, 0);
                new_state.exp_avg_sq = keep_exp_avg_sq.cat(zeros, 0);
                // CRITICAL: Update size to match new parameter count
                size_t new_size = num_keep + num_split * split_size;
                new_state.size = new_size;
                LOG_DEBUG("  New state.size = {}", new_state.size);
                LOG_DEBUG("  exp_avg shape after cat: {}", new_state.exp_avg.shape().str());
                _optimizer->set_state(param_type, new_state);
            }
        };

        update_optimizer_state(ParamType::Means);
        update_optimizer_state(ParamType::Rotation);
        update_optimizer_state(ParamType::Scaling);
        update_optimizer_state(ParamType::Sh0);
        update_optimizer_state(ParamType::ShN);
        update_optimizer_state(ParamType::Opacity);
    }

    void DefaultStrategy::grow_gs(int iter) {
        lfs::core::Tensor numer = _splat_data._densification_info[1];
        lfs::core::Tensor denom = _splat_data._densification_info[0];
        const lfs::core::Tensor grads = numer / denom.clamp_min(1.0f);

        const lfs::core::Tensor is_grad_high = grads > _params->grad_threshold;

        // Get max along last dimension
        const lfs::core::Tensor max_values = _splat_data.get_scaling().max(-1, false);
        const lfs::core::Tensor is_small = max_values <= _params->grow_scale3d * _splat_data.get_scene_scale();
        const lfs::core::Tensor is_duplicated = is_grad_high.logical_and(is_small);

        const auto num_duplicates = static_cast<int64_t>(is_duplicated.sum_scalar());

        const lfs::core::Tensor is_large = is_small.logical_not();
        lfs::core::Tensor is_split = is_grad_high.logical_and(is_large);
        const auto num_split = static_cast<int64_t>(is_split.sum_scalar());

        // First duplicate
        if (num_duplicates > 0) {
            duplicate(is_duplicated);
        }

        // New Gaussians added by duplication will not be split
        auto zeros_to_concat = lfs::core::Tensor::zeros_bool({static_cast<size_t>(num_duplicates)}, is_split.device());
        is_split = is_split.cat(zeros_to_concat, 0);

        if (num_split > 0) {
            split(is_split);
        }
    }

    void DefaultStrategy::remove(const lfs::core::Tensor& is_prune) {
        const lfs::core::Tensor sampled_idxs = is_prune.logical_not().nonzero().squeeze(-1);

        const auto param_fn = [&sampled_idxs](const int i, const lfs::core::Tensor& param) {
            return param.index_select(0, sampled_idxs);
        };

        const auto optimizer_fn = [&sampled_idxs](
            AdamParamState& state,
            const lfs::core::Tensor& new_param) {
            // For remove, we select only the surviving Gaussians' optimizer state
            size_t old_size = state.size;
            state.exp_avg = state.exp_avg.index_select(0, sampled_idxs);
            state.exp_avg_sq = state.exp_avg_sq.index_select(0, sampled_idxs);
            // CRITICAL: Update size to match new parameter count
            size_t new_size = sampled_idxs.shape()[0];
            state.size = new_size;
            LOG_DEBUG("  remove() size update: {} -> {} (removed {})", old_size, new_size, old_size - new_size);
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);
    }

    void DefaultStrategy::prune_gs(int iter) {
        // Check for low opacity
        lfs::core::Tensor is_prune = _splat_data.get_opacity() < _params->prune_opacity;

        auto rotation_raw = _splat_data.rotation_raw();
        is_prune = is_prune.logical_or((rotation_raw * rotation_raw).sum(-1, false) < 1e-8f);

        // Check for too large Gaussians
        if (iter > _params->reset_every) {
            const lfs::core::Tensor max_values = _splat_data.get_scaling().max(-1, false);
            lfs::core::Tensor is_too_big = max_values > _params->prune_scale3d * _splat_data.get_scene_scale();
            is_prune = is_prune.logical_or(is_too_big);
        }

        const auto num_prunes = static_cast<int64_t>(is_prune.sum_scalar());
        if (num_prunes > 0) {
            remove(is_prune);
        }
    }

    void DefaultStrategy::reset_opacity() {
        const auto threshold = 2.0f * _params->prune_opacity;

        const auto param_fn = [&threshold](const int i, const lfs::core::Tensor& param) {
            if (i == 5) {
                // For opacity parameter, clamp to logit(threshold)
                const float logit_threshold = std::log(threshold / (1.0f - threshold));
                return param.clamp_max(logit_threshold);
            }
            LOG_ERROR("Invalid parameter index for reset_opacity: {}", i);
            return param;
        };

        const auto optimizer_fn = [](AdamParamState& state, const lfs::core::Tensor& new_param) {
            // Reset optimizer state for opacity to zeros
            state.exp_avg = lfs::core::Tensor::zeros_like(state.exp_avg);
            state.exp_avg_sq = lfs::core::Tensor::zeros_like(state.exp_avg_sq);
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data, {5});
    }

    void DefaultStrategy::post_backward(int iter, RenderOutput& render_output) {
        // Increment SH degree every 1000 iterations
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data.increment_sh_degree();
        }

        if (iter == _params->stop_refine) {
            // Reset densification info at the end of refinement. Saves memory and processing time.
            _splat_data._densification_info = lfs::core::Tensor::empty({0});
        }

        if (iter >= _params->stop_refine) {
            return;
        }

        if (is_refining(iter)) {
            grow_gs(iter);
            prune_gs(iter);

            _splat_data._densification_info = lfs::core::Tensor::zeros(
                {2, static_cast<size_t>(_splat_data.size())},
                _splat_data.means().device());
        }

        if (iter % _params->reset_every == 0 && iter > 0) {
            reset_opacity();
        }
    }

    void DefaultStrategy::step(int iter) {
        if (iter < _params->iterations) {
            _optimizer->step(iter);
            _optimizer->zero_grad(iter);
            _scheduler->step();
        }
    }
} // namespace lfs::training
