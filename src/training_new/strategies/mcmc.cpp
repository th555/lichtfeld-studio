/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcmc.hpp"
#include "kernels/mcmc_kernels.hpp"
#include "strategy_utils.hpp"
#include "core_new/logger.hpp"
#include "core_new/tensor/internal/memory_pool.hpp"
#include <cmath>

namespace lfs::training {

    MCMC::MCMC(lfs::core::SplatData&& splat_data)
        : _splat_data(std::move(splat_data)) {
    }

    lfs::core::Tensor MCMC::multinomial_sample(const lfs::core::Tensor& weights, int n, bool replacement) {
        // Use the tensor library's built-in multinomial sampling
        return lfs::core::Tensor::multinomial(weights, n, replacement);
    }

    void MCMC::update_optimizer_for_relocate(
        const lfs::core::Tensor& sampled_indices,
        const lfs::core::Tensor& dead_indices,
        ParamType param_type) {

        // Reset optimizer state (exp_avg and exp_avg_sq) for relocated Gaussians
        // Use GPU version for efficiency (indices already on GPU)
        _optimizer->relocate_params_at_indices_gpu(
            param_type,
            sampled_indices.ptr<int64_t>(),
            sampled_indices.numel()
        );
    }

    int MCMC::relocate_gs() {
        LOG_TIMER("MCMC::relocate_gs");
        using namespace lfs::core;

        // Get opacities (handle both [N] and [N, 1] shapes)
        Tensor opacities;
        {
            LOG_TIMER("relocate_get_opacities");
            opacities = _splat_data.get_opacity();
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
                opacities = opacities.squeeze(-1);
            }
        }

        // Find dead Gaussians: opacity <= min_opacity OR rotation magnitude near zero
        Tensor dead_mask, dead_indices;
        int n_dead;
        {
            LOG_TIMER("relocate_find_dead");
            Tensor rotation_raw = _splat_data.rotation_raw();
            Tensor rot_mag_sq = (rotation_raw * rotation_raw).sum(-1);
            dead_mask = (opacities <= _params->min_opacity).logical_or(rot_mag_sq < 1e-8f);
            dead_indices = dead_mask.nonzero().squeeze(-1);
            n_dead = dead_indices.numel();
        }

        if (n_dead == 0)
            return 0;

        Tensor alive_indices;
        {
            LOG_TIMER("relocate_find_alive");
            Tensor alive_mask = dead_mask.logical_not();
            alive_indices = alive_mask.nonzero().squeeze(-1);
        }

        if (alive_indices.numel() == 0)
            return 0;

        // Sample from alive Gaussians based on opacity
        Tensor sampled_idxs;
        {
            LOG_TIMER("relocate_multinomial_sample");
            Tensor probs = opacities.index_select(0, alive_indices);
            Tensor sampled_idxs_local = multinomial_sample(probs, n_dead, true);
            sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);
        }

        // Get parameters for sampled Gaussians
        Tensor sampled_opacities, sampled_scales;
        {
            LOG_TIMER("relocate_get_sampled_params");
            // Use fused gather kernel - CRITICAL: ensure tensors are contiguous!
            const size_t n_samples = sampled_idxs.numel();
            const size_t N = opacities.numel();

            // Get source tensors and ensure they're contiguous
            Tensor opacities_contig = opacities.contiguous();
            Tensor scales = _splat_data.get_scaling().contiguous();

            // Allocate outputs
            sampled_opacities = Tensor::empty({n_samples}, Device::CUDA, DataType::Float32);
            sampled_scales = Tensor::empty({n_samples, 3}, Device::CUDA, DataType::Float32);

            // Launch fused kernel
            mcmc::launch_gather_2tensors(
                sampled_idxs.ptr<int64_t>(),
                opacities_contig.ptr<float>(),
                scales.ptr<float>(),
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                n_samples,
                1,  // dim_a: opacities are [N]
                3,  // dim_b: scales are [N, 3]
                N
            );
        }

        // Count occurrences of each sampled index (how many times each was sampled)
        Tensor ratios;
        {
            LOG_TIMER("relocate_count_occurrences");
            ratios = Tensor::ones_like(opacities, DataType::Int32);
            ratios = ratios.index_add_(0, sampled_idxs, Tensor::ones({sampled_idxs.numel()}, Device::CUDA, DataType::Int32));
            ratios = ratios.index_select(0, sampled_idxs).contiguous();

            // Clamp ratios to [1, n_max]
            const int n_max = static_cast<int>(_binoms.shape()[0]);
            ratios = ratios.clamp(1, n_max);
        }

        // Allocate output tensors and call CUDA kernel
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("relocate_cuda_kernel");
            const int n_max = static_cast<int>(_binoms.shape()[0]);
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);

            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                _binoms.ptr<float>(),
                n_max,
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel()
            );
        }

        // Clamp new opacities and compute raw values
        Tensor new_opacity_raw;
        {
            LOG_TIMER("relocate_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = (new_opacities / (Tensor::ones_like(new_opacities) - new_opacities)).log();

            if (_splat_data.opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

        // Update parameters
        {
            LOG_TIMER("relocate_update_params");
            const int opacity_dim = (_splat_data.opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data.means().shape()[0];  // Total number of Gaussians

            // Compute log(scales) for the new scales
            Tensor new_scales_log = new_scales.log();

            // Update sampled indices with new opacity/scaling using direct CUDA kernel
            // This preserves tensor capacity (unlike index_put_ which creates new tensors)
            mcmc::launch_update_scaling_opacity(
                sampled_idxs.ptr<int64_t>(),
                new_scales_log.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data.scaling_raw().ptr<float>(),
                _splat_data.opacity_raw().ptr<float>(),
                sampled_idxs.numel(),
                opacity_dim,
                N
            );

            // Then copy from sampled indices to dead indices using fused copy kernel
            mcmc::launch_copy_gaussian_params(
                sampled_idxs.ptr<int64_t>(),
                dead_indices.ptr<int64_t>(),
                _splat_data.means().ptr<float>(),
                _splat_data.sh0().ptr<float>(),
                _splat_data.shN().ptr<float>(),
                _splat_data.scaling_raw().ptr<float>(),
                _splat_data.rotation_raw().ptr<float>(),
                _splat_data.opacity_raw().ptr<float>(),
                dead_indices.numel(),
                _splat_data.shN().shape()[1],
                opacity_dim,
                N  // Pass N for bounds checking
            );
        }

        // Update optimizer states for all parameters
        {
            LOG_TIMER("relocate_update_optimizer");
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Means);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Sh0);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::ShN);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Scaling);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Rotation);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Opacity);
        }

        return n_dead;
    }

    int MCMC::add_new_gs() {
        LOG_TIMER("MCMC::add_new_gs");
        using namespace lfs::core;

        if (!_optimizer) {
            LOG_ERROR("MCMC::add_new_gs: optimizer not initialized");
            return 0;
        }

        const int current_n = _splat_data.size();
        const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
        const int n_new = std::max(0, n_target - current_n);

        if (n_new == 0)
            return 0;

        // Get opacities (handle both [N] and [N, 1] shapes)
        Tensor opacities;
        {
            LOG_TIMER("add_new_get_opacities");
            opacities = _splat_data.get_opacity();
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
                opacities = opacities.squeeze(-1);
            }
        }

        Tensor sampled_idxs;
        {
            LOG_TIMER("add_new_multinomial_sample");
            auto probs = opacities.flatten();
            sampled_idxs = multinomial_sample(probs, n_new, true);
        }

        // Get parameters for sampled Gaussians
        Tensor sampled_opacities, sampled_scales;
        {
            LOG_TIMER("add_new_get_sampled_params");
            sampled_opacities = opacities.index_select(0, sampled_idxs);
            sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);
        }

        // Count occurrences (ratio starts at 0, add 1 for each occurrence, then add 1 more)
        Tensor ratios;
        {
            LOG_TIMER("add_new_count_occurrences");
            ratios = Tensor::zeros({opacities.numel()}, Device::CUDA, DataType::Float32);
            ratios = ratios.index_add_(0, sampled_idxs, Tensor::ones({sampled_idxs.numel()}, Device::CUDA));
            ratios = ratios.index_select(0, sampled_idxs) + Tensor::ones_like(ratios.index_select(0, sampled_idxs));

            // Clamp and convert to int32
            const int n_max = static_cast<int>(_binoms.shape()[0]);
            ratios = ratios.clamp(1.0f, static_cast<float>(n_max));
            ratios = ratios.to(DataType::Int32).contiguous();
        }

        // Allocate output tensors and call CUDA kernel
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("add_new_relocation_kernel");
            const int n_max = static_cast<int>(_binoms.shape()[0]);
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);

            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                _binoms.ptr<float>(),
                n_max,
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel()
            );
        }

        // Clamp new opacities and prepare raw values
        Tensor new_opacity_raw, new_scaling_raw;
        {
            LOG_TIMER("add_new_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = (new_opacities / (Tensor::ones_like(new_opacities) - new_opacities)).log();
            new_scaling_raw = new_scales.log();

            if (_splat_data.opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

        // CRITICAL: Update existing Gaussians FIRST (before concatenation)
        // This matches the legacy implementation and ensures the correct values are copied
        {
            LOG_TIMER("add_new_update_original");
            const int opacity_dim = (_splat_data.opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data.means().shape()[0];

            // Use direct CUDA kernel to preserve tensor capacity (unlike index_put_ which creates new tensors)
            mcmc::launch_update_scaling_opacity(
                sampled_idxs.ptr<int64_t>(),
                new_scaling_raw.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data.scaling_raw().ptr<float>(),
                _splat_data.opacity_raw().ptr<float>(),
                sampled_idxs.numel(),
                opacity_dim,
                N
            );
        }

        // Prepare new Gaussians to concatenate (gather updated values)
        Tensor new_means, new_sh0, new_shN, new_rotation, new_opacity_to_add, new_scaling_to_add;
        {
            LOG_TIMER("add_new_gather_params");
            // Gather parameters for new Gaussians
            // NOTE: We must gather opacity/scaling AFTER updating them above!
            new_means = _splat_data.means().index_select(0, sampled_idxs);
            new_sh0 = _splat_data.sh0().index_select(0, sampled_idxs);
            new_shN = _splat_data.shN().index_select(0, sampled_idxs);
            new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs);
            new_opacity_to_add = _splat_data.opacity_raw().index_select(0, sampled_idxs);
            new_scaling_to_add = _splat_data.scaling_raw().index_select(0, sampled_idxs);
        }

        // Concatenate all parameters using optimizer's add_new_params
        {
            LOG_TIMER("add_new_concatenate_params");
            // Note: add_new_params REPLACES the parameter tensors with concatenated versions
            _optimizer->add_new_params(ParamType::Means, new_means);
            _optimizer->add_new_params(ParamType::Sh0, new_sh0);
            _optimizer->add_new_params(ParamType::ShN, new_shN);
            _optimizer->add_new_params(ParamType::Scaling, new_scaling_to_add);
            _optimizer->add_new_params(ParamType::Rotation, new_rotation);
            _optimizer->add_new_params(ParamType::Opacity, new_opacity_to_add);
        }

        return n_new;
    }

    // Test helper: add_new_gs with explicitly specified indices (no multinomial sampling)
    int MCMC::add_new_gs_with_indices_test(const lfs::core::Tensor& sampled_idxs) {
        LOG_TIMER("MCMC::add_new_gs_with_indices_test");
        using namespace lfs::core;

        if (!_optimizer) {
            LOG_ERROR("add_new_gs_with_indices_test called but optimizer not initialized");
            return 0;
        }

        const int n_new = sampled_idxs.numel();
        if (n_new == 0)
            return 0;

        // Ensure indices are Int64 (test may pass Int32)
        Tensor sampled_idxs_i64 = (sampled_idxs.dtype() == DataType::Int64) ? sampled_idxs : sampled_idxs.to(DataType::Int64);

        // Get opacities
        auto opacities = _splat_data.get_opacity();

        // Get parameters for sampled Gaussians
        auto sampled_opacities = opacities.index_select(0, sampled_idxs_i64);
        auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs_i64);

        // Count occurrences
        auto ratios = Tensor::zeros({static_cast<size_t>(_splat_data.size())}, Device::CUDA, DataType::Float32);
        ratios.index_add_(0, sampled_idxs_i64, Tensor::ones_like(sampled_idxs_i64).to(DataType::Float32));
        ratios = ratios.index_select(0, sampled_idxs_i64) + 1.0f;

        // Clamp and convert to int
        const int n_max = static_cast<int>(_binoms.shape()[0]);
        ratios = ratios.clamp(1.0f, static_cast<float>(n_max));
        ratios = ratios.to(DataType::Int32).contiguous();

        // Call the CUDA relocation function
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("add_new_relocation");
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);

            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                _binoms.ptr<float>(),
                n_max,
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel()
            );
        }

        // Clamp new opacities and prepare raw values
        Tensor new_opacity_raw, new_scaling_raw;
        {
            LOG_TIMER("add_new_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = (new_opacities / (Tensor::ones_like(new_opacities) - new_opacities)).log();
            new_scaling_raw = new_scales.log();

            if (_splat_data.opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

        // CRITICAL: Update existing Gaussians FIRST (before concatenation)
        // This matches the legacy implementation and ensures the correct values are copied
        {
            LOG_TIMER("add_new_update_original");
            const int opacity_dim = (_splat_data.opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data.means().shape()[0];

            // Use direct CUDA kernel to preserve tensor capacity (unlike index_put_ which creates new tensors)
            mcmc::launch_update_scaling_opacity(
                sampled_idxs_i64.ptr<int64_t>(),
                new_scaling_raw.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data.scaling_raw().ptr<float>(),
                _splat_data.opacity_raw().ptr<float>(),
                sampled_idxs_i64.numel(),
                opacity_dim,
                N
            );
        }

        // Use fused append_gather() operation - NO intermediate allocations!
        {
            LOG_TIMER("add_new_params_gather");
            // NOTE: We must gather opacity/scaling AFTER updating them above!
            _optimizer->add_new_params_gather(ParamType::Means, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Sh0, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::ShN, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Rotation, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Opacity, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Scaling, sampled_idxs_i64);
        }

        return n_new;
    }

    void MCMC::inject_noise() {
        LOG_TIMER("MCMC::inject_noise");
        using namespace lfs::core;

        // Get current learning rate from optimizer (after scheduler has updated it)
        const float current_lr = _optimizer->get_lr() * _noise_lr;

        // Generate noise
        Tensor noise;
        {
            LOG_TIMER("inject_noise_generate");
            noise = Tensor::randn_like(_splat_data.means());
        }

        // Call CUDA add_noise kernel
        {
            LOG_TIMER("inject_noise_cuda_kernel");
            mcmc::launch_add_noise_kernel(
                _splat_data.opacity_raw().ptr<float>(),
                _splat_data.scaling_raw().ptr<float>(),
                _splat_data.rotation_raw().ptr<float>(),
                noise.ptr<float>(),
                _splat_data.means().ptr<float>(),
                current_lr,
                _splat_data.size()
            );
        }
    }

    void MCMC::post_backward(int iter, RenderOutput& render_output) {
        LOG_TIMER("MCMC::post_backward");

        // Increment SH degree every sh_degree_interval iterations
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data.increment_sh_degree();
        }

        // Refine Gaussians
        if (is_refining(iter)) {
            // Relocate dead Gaussians
            int n_relocated = relocate_gs();
            if (n_relocated > 0) {
                LOG_DEBUG("MCMC: Relocated {} dead Gaussians at iteration {}", n_relocated, iter);
            }

            // Add new Gaussians
            int n_added = add_new_gs();
            if (n_added > 0) {
                LOG_DEBUG("MCMC: Added {} new Gaussians at iteration {} (total: {})",
                         n_added, iter, _splat_data.size());
            }
            // Release cached memory from cudaMallocAsync pool to avoid memory bloat
            // This is especially important after add_new_gs() which creates many temporary tensors
            lfs::core::CudaMemoryPool::instance().trim_cached_memory();
        }

        // Inject noise to positions every iteration
        inject_noise();
    }

    void MCMC::step(int iter) {
        LOG_TIMER("MCMC::step");
        if (iter < _params->iterations) {
            {
                LOG_TIMER("step_optimizer_step");
                _optimizer->step(iter);
            }
            {
                LOG_TIMER("step_zero_grad");
                _optimizer->zero_grad(iter);
            }
            {
                LOG_TIMER("step_scheduler");
                _scheduler->step();
            }
        }
    }

    void MCMC::remove_gaussians(const lfs::core::Tensor& mask) {
        using namespace lfs::core;

        // Convert bool to int32 for sum
        Tensor mask_int = mask.to(DataType::Int32);
        int n_remove = mask_int.sum().item<int>();

        LOG_INFO("MCMC::remove_gaussians called: mask size={}, n_remove={}, current size={}",
                 mask.numel(), n_remove, _splat_data.size());

        if (n_remove == 0) {
            LOG_DEBUG("MCMC: No Gaussians to remove");
            return;
        }

        LOG_DEBUG("MCMC: Removing {} Gaussians", n_remove);

        // Get indices to keep
        Tensor keep_mask = mask.logical_not();
        Tensor keep_indices = keep_mask.nonzero().squeeze(-1);

        // Select only the Gaussians we want to keep
        _splat_data.means() = _splat_data.means().index_select(0, keep_indices).contiguous();
        _splat_data.sh0() = _splat_data.sh0().index_select(0, keep_indices).contiguous();
        _splat_data.shN() = _splat_data.shN().index_select(0, keep_indices).contiguous();
        _splat_data.scaling_raw() = _splat_data.scaling_raw().index_select(0, keep_indices).contiguous();
        _splat_data.rotation_raw() = _splat_data.rotation_raw().index_select(0, keep_indices).contiguous();
        _splat_data.opacity_raw() = _splat_data.opacity_raw().index_select(0, keep_indices).contiguous();

        // Recreate optimizer with reduced parameters
        // This is simpler than trying to manually update optimizer state
        _optimizer = create_optimizer(_splat_data, *_params);

        // Recreate scheduler
        const double gamma = std::pow(0.01, 1.0 / _params->iterations);
        _scheduler = create_scheduler(*_params, *_optimizer);
    }

    void MCMC::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        using namespace lfs::core;

        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        // Pre-allocate tensor capacity if max_cap is specified
        if (_params->max_cap > 0) {
            const size_t capacity = static_cast<size_t>(_params->max_cap);
            const size_t current_size = _splat_data.size();
            LOG_INFO("Pre-allocating capacity for {} Gaussians (current size: {}, utilization: {:.1f}%)",
                     capacity, current_size, 100.0f * current_size / capacity);

            try {
                // Pre-allocate attribute tensors
                LOG_DEBUG("  Reserving capacity for parameters:");
                _splat_data.means().reserve(capacity);
                LOG_DEBUG("    means: size={}, capacity={}", _splat_data.means().shape()[0], _splat_data.means().capacity());
                _splat_data.sh0().reserve(capacity);
                LOG_DEBUG("    sh0: size={}, capacity={}", _splat_data.sh0().shape()[0], _splat_data.sh0().capacity());
                _splat_data.shN().reserve(capacity);
                LOG_DEBUG("    shN: size={}, capacity={}", _splat_data.shN().shape()[0], _splat_data.shN().capacity());
                _splat_data.scaling_raw().reserve(capacity);
                LOG_DEBUG("    scaling: size={}, capacity={}", _splat_data.scaling_raw().shape()[0], _splat_data.scaling_raw().capacity());
                _splat_data.rotation_raw().reserve(capacity);
                LOG_DEBUG("    rotation: size={}, capacity={}", _splat_data.rotation_raw().shape()[0], _splat_data.rotation_raw().capacity());
                _splat_data.opacity_raw().reserve(capacity);
                LOG_DEBUG("    opacity: size={}, capacity={}", _splat_data.opacity_raw().shape()[0], _splat_data.opacity_raw().capacity());

                // Pre-allocate gradient tensors (must allocate gradients first)
                if (!_splat_data.has_gradients()) {
                    _splat_data.allocate_gradients();
                }
                LOG_DEBUG("  Reserving capacity for gradients:");
                _splat_data.means_grad().reserve(capacity);
                LOG_DEBUG("    means_grad: size={}, capacity={}", _splat_data.means_grad().shape()[0], _splat_data.means_grad().capacity());
                _splat_data.sh0_grad().reserve(capacity);
                LOG_DEBUG("    sh0_grad: size={}, capacity={}", _splat_data.sh0_grad().shape()[0], _splat_data.sh0_grad().capacity());
                _splat_data.shN_grad().reserve(capacity);
                LOG_DEBUG("    shN_grad: size={}, capacity={}", _splat_data.shN_grad().shape()[0], _splat_data.shN_grad().capacity());
                _splat_data.scaling_grad().reserve(capacity);
                LOG_DEBUG("    scaling_grad: size={}, capacity={}", _splat_data.scaling_grad().shape()[0], _splat_data.scaling_grad().capacity());
                _splat_data.rotation_grad().reserve(capacity);
                LOG_DEBUG("    rotation_grad: size={}, capacity={}", _splat_data.rotation_grad().shape()[0], _splat_data.rotation_grad().capacity());
                _splat_data.opacity_grad().reserve(capacity);
                LOG_DEBUG("    opacity_grad: size={}, capacity={}", _splat_data.opacity_grad().shape()[0], _splat_data.opacity_grad().capacity());

                LOG_INFO("✓ Tensor capacity pre-allocation complete: {}/{} Gaussians ({:.1f}% utilization)",
                         current_size, capacity, 100.0f * current_size / capacity);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to pre-allocate capacity: {}. Continuing without pre-allocation.", e.what());
            }
        }

        // Initialize binomial coefficients (same as original)
        const int n_max = 51;
        std::vector<float> binoms_data(n_max * n_max, 0.0f);
        for (int n = 0; n < n_max; ++n) {
            for (int k = 0; k <= n; ++k) {
                float binom = 1.0f;
                for (int i = 0; i < k; ++i) {
                    binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
                }
                binoms_data[n * n_max + k] = binom;
            }
        }
        _binoms = Tensor::from_vector(binoms_data, TensorShape({static_cast<size_t>(n_max), static_cast<size_t>(n_max)}), Device::CUDA);

        // Initialize optimizer using strategy_utils helper
        _optimizer = create_optimizer(_splat_data, *_params);

        // Initialize scheduler
        _scheduler = create_scheduler(*_params, *_optimizer);

        LOG_INFO("MCMC strategy initialized with {} Gaussians", _splat_data.size());
    }

    bool MCMC::is_refining(int iter) const {
        return (iter < _params->stop_refine &&
                iter > _params->start_refine &&
                iter % _params->refine_every == 0);
    }

} // namespace lfs::training
