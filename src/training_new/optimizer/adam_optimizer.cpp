/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_optimizer.hpp"
#include "adam_api.h"  // fast_lfs::optimizer::adam_step_raw
#include "core_new/logger.hpp"
#include <cmath>
#include <stdexcept>

namespace lfs::training {

    AdamOptimizer::AdamOptimizer(lfs::core::SplatData& splat_data, const AdamConfig& config)
        : splat_data_(splat_data), config_(config) {

        LOG_DEBUG("AdamOptimizer constructor: config.initial_capacity={}, config.growth_factor={}",
                  config_.initial_capacity, config_.growth_factor);

        // Ensure gradients are allocated
        if (!splat_data_.has_gradients()) {
            splat_data_.allocate_gradients();
            LOG_DEBUG("Allocated gradients for optimizer");
        }
    }

    void AdamOptimizer::step(int iteration) {
        // Optimize each parameter
        step_param(ParamType::Means, iteration);
        step_param(ParamType::Sh0, iteration);
        step_param(ParamType::ShN, iteration);
        step_param(ParamType::Scaling, iteration);
        step_param(ParamType::Rotation, iteration);
        step_param(ParamType::Opacity, iteration);
    }

    void AdamOptimizer::zero_grad(int iteration) {
        // TODO: Optional - Skip SH gradients on certain iterations (matching old behavior)
        // For now, just zero everything
        splat_data_.zero_gradients();
    }

    lfs::core::Tensor& AdamOptimizer::get_param(ParamType type) {
        switch (type) {
            case ParamType::Means: return splat_data_.means();
            case ParamType::Sh0: return splat_data_.sh0();
            case ParamType::ShN: return splat_data_.shN();
            case ParamType::Scaling: return splat_data_.scaling_raw();
            case ParamType::Rotation: return splat_data_.rotation_raw();
            case ParamType::Opacity: return splat_data_.opacity_raw();
        }
        throw std::runtime_error("Invalid param type");
    }

    lfs::core::Tensor& AdamOptimizer::get_grad(ParamType type) {
        switch (type) {
            case ParamType::Means: return splat_data_.means_grad();
            case ParamType::Sh0: return splat_data_.sh0_grad();
            case ParamType::ShN: return splat_data_.shN_grad();
            case ParamType::Scaling: return splat_data_.scaling_grad();
            case ParamType::Rotation: return splat_data_.rotation_grad();
            case ParamType::Opacity: return splat_data_.opacity_grad();
        }
        throw std::runtime_error("Invalid param type");
    }

    std::string AdamOptimizer::param_name(ParamType type) const {
        switch (type) {
            case ParamType::Means: return "means";
            case ParamType::Sh0: return "sh0";
            case ParamType::ShN: return "shN";
            case ParamType::Scaling: return "scaling";
            case ParamType::Rotation: return "rotation";
            case ParamType::Opacity: return "opacity";
        }
        return "unknown";
    }

    void AdamOptimizer::init_state(ParamType type) {
        auto& param = get_param(type);
        auto name = param_name(type);

        // Validate param before creating state
        if (!param.is_valid()) {
            throw std::runtime_error("init_state: parameter " + name + " is not valid!");
        }
        if (param.ndim() == 0) {
            throw std::runtime_error("init_state: parameter " + name + " has rank 0! This will create rank-0 optimizer state.");
        }

        auto& state = states_[name];
        size_t param_size = param.shape()[0];

        // Calculate initial capacity with pre-allocation if configured
        size_t initial_cap = compute_new_capacity(0, param_size);

        // STRATEGY: Use reserve() to pre-allocate outside the pool (direct cudaMalloc)
        // This keeps optimizer state in persistent memory, not in the cudaMallocAsync pool
        // which allows the pool to be trimmed for temporary allocations

        // Create minimal tensors with actual size
        state.exp_avg = lfs::core::Tensor::zeros(param.shape(), param.device());
        state.exp_avg_sq = lfs::core::Tensor::zeros(param.shape(), param.device());

        // Pre-allocate capacity if requested (uses direct cudaMalloc, not pool)
        if (initial_cap > param_size) {
            state.exp_avg.reserve(initial_cap);
            state.exp_avg_sq.reserve(initial_cap);
            state.capacity = initial_cap;
            state.size = param_size;

            LOG_INFO("Initialized optimizer state for {} with reserve() (size: {}, capacity: {}) - outside pool",
                      name, param_size, initial_cap);
        } else {
            // No pre-allocation: exact fit
            state.capacity = param_size;
            state.size = param_size;

            LOG_DEBUG("Initialized optimizer state for {} (size: {}, no pre-allocation)", name, param_size);
        }

        state.step_count = 0;
    }

    void AdamOptimizer::step_param(ParamType type, int iteration) {
        auto& param = get_param(type);
        auto& grad = get_grad(type);

        // Skip if no gradient or if gradient is all zeros (not yet computed)
        if (!grad.is_valid() || grad.numel() == 0) {
            return;
        }

        // OPTIMIZATION: Skip if parameter doesn't exist yet (lazy initialization)
        if (!param.is_valid() || param.numel() == 0) {
            return;
        }

        auto name = param_name(type);

        // Initialize state on first call
        if (states_.find(name) == states_.end()) {
            init_state(type);
        }

        auto& state = states_[name];
        state.step_count++;

        // Compute bias correction factors
        float bias_correction1_rcp = 1.0f / (1.0f - std::pow(config_.beta1, state.step_count));
        float bias_correction2_sqrt_rcp = 1.0f / std::sqrt(1.0f - std::pow(config_.beta2, state.step_count));

        // Get per-parameter learning rate
        float param_lr = get_param_lr(type);

        // When fast path is used in extend_state_for_new_params, state tensors have
        // excess capacity (state.exp_avg.shape()[0] > param.shape()[0]).
        // We must ensure we only operate on the valid elements.

        size_t param_size = param.shape()[0];
        size_t state_size = state.size;

        // CRITICAL: Verify param and state are synchronized
        if (param_size != state_size) {
            LOG_ERROR("  BUG: param size ({}) != state.size ({})", param_size, state_size);
            LOG_ERROR("  This indicates add_new_params and extend_state_for_new_params are out of sync!");
            LOG_ERROR("  Parameter: {}", name);
            throw std::runtime_error("Optimizer state desynchronization detected");
        }

        // Calculate number of elements to process
        // This MUST use state.size, not param.shape()[0], because after fast path:
        //   - param.shape()[0] reflects actual data size
        //   - state.size tracks logical size (should match param)
        //   - state.exp_avg.shape()[0] may be larger (excess capacity)
        size_t feature_dim = param.numel() / param_size;  // e.g., 3 for means, 4 for rotation
        size_t num_elements = state_size * feature_dim;

        // Optional diagnostic logging (only at trace level)
        if (iteration % 1000 == 0 && state.capacity > state.size) {
            LOG_TRACE("Optimizer capacity usage for {}: {}/{} ({:.1f}%)",
                     name, state_size, state.capacity, 100.0f * state_size / state.capacity);
        }

        // Call fused CUDA kernel - operates ONLY on valid elements
        // Uses state.size * feature_dim instead of param.numel()
        fast_lfs::optimizer::adam_step_raw(
            param.ptr<float>(),
            state.exp_avg.ptr<float>(),
            state.exp_avg_sq.ptr<float>(),
            grad.ptr<float>(),
            static_cast<int>(num_elements),  // Based on state.size!
            param_lr,  // Use per-parameter learning rate
            config_.beta1,
            config_.beta2,
            config_.eps,
            bias_correction1_rcp,
            bias_correction2_sqrt_rcp
        );
    }

    void AdamOptimizer::reset_state_at_indices(ParamType type, const std::vector<int64_t>& indices) {
        auto name = param_name(type);

        // Ensure state exists
        if (states_.find(name) == states_.end()) {
            LOG_DEBUG("State for {} not initialized yet, skipping reset", name);
            return;
        }

        if (indices.empty()) {
            return;  // Nothing to do
        }

        auto& state = states_[name];

        // Calculate row size (product of all dimensions except first)
        auto state_shape = state.exp_avg.shape();
        int row_size = 1;
        for (size_t i = 1; i < state_shape.rank(); i++) {
            row_size *= state_shape[i];
        }

        // Allocate GPU memory for indices and copy from host
        int64_t* indices_device_ptr;
        cudaMalloc(&indices_device_ptr, indices.size() * sizeof(int64_t));
        cudaMemcpy(indices_device_ptr, indices.data(), indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

        // Use batched CUDA kernel for much better performance (600x faster!)
        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg.template ptr<float>(),
            indices_device_ptr,
            indices.size(),
            row_size
        );

        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg_sq.template ptr<float>(),
            indices_device_ptr,
            indices.size(),
            row_size
        );

        cudaFree(indices_device_ptr);

        LOG_DEBUG("Reset optimizer state for {} at {} indices (batched GPU kernel)", name, indices.size());
    }

    void AdamOptimizer::extend_state_for_new_params(ParamType type, size_t n_new) {
        auto& param = get_param(type);
        auto name = param_name(type);

        // Ensure state exists
        if (states_.find(name) == states_.end()) {
            // If state doesn't exist yet, it will be initialized on first step
            LOG_DEBUG("State for {} not initialized yet, will be initialized on first step", name);
            return;
        }

        auto& state = states_[name];
        size_t new_size = state.size + n_new;

        // DISABLED: Fast path optimization is buggy - tensors need to be resized, not just state.size updated
        // TODO: Implement proper fast path using narrow() or a resize_bytes() operation
        /*
        if (new_size <= state.capacity) {
            // Fast path: would avoid reallocation IF we could resize tensors properly
            ...
            return;
        }
        */

        // Slow path: Need to grow capacity
        auto param_shape = param.shape();

        // Defensive check: ensure param is valid
        if (!param.is_valid() || param_shape.rank() == 0) {
            throw std::runtime_error("extend_state_for_new_params: param " + name +
                                   " is invalid! is_valid=" + std::to_string(param.is_valid()) +
                                   ", rank=" + std::to_string(param_shape.rank()));
        }

        // Defensive check: ensure state tensors are valid
        if (!state.exp_avg.is_valid() || state.exp_avg.ndim() == 0) {
            throw std::runtime_error("extend_state_for_new_params: state.exp_avg for " + name +
                                   " is invalid! is_valid=" + std::to_string(state.exp_avg.is_valid()) +
                                   ", ndim=" + std::to_string(state.exp_avg.ndim()));
        }

        // SIMPLE FIX: Just create zeros with the exact size needed (n_new params)
        // No capacity pre-allocation optimization - not worth the complexity without proper slice operations
        std::vector<size_t> new_zeros_dims(param_shape.rank());
        new_zeros_dims[0] = n_new;
        for (size_t i = 1; i < param_shape.rank(); i++) {
            new_zeros_dims[i] = param_shape[i];
        }
        auto new_zeros_shape = lfs::core::TensorShape(new_zeros_dims);
        auto new_zeros = lfs::core::Tensor::zeros(new_zeros_shape, param.device());

        // Concatenate: old state + new zeros
        // NOTE: state.exp_avg may have shape larger than state.size due to previous pre-allocation
        // We need to handle this by only using the first state.size elements
        // WORKAROUND: Since we don't have narrow/slice, recreate tensors with exact size
        std::vector<size_t> current_dims(param_shape.rank());
        current_dims[0] = state.size;
        for (size_t i = 1; i < param_shape.rank(); i++) {
            current_dims[i] = param_shape[i];
        }
        auto current_shape = lfs::core::TensorShape(current_dims);

        // Copy first state.size elements (TODO: optimize with slice when available)
        auto exp_avg_trimmed = lfs::core::Tensor::zeros(current_shape, param.device());
        auto exp_avg_sq_trimmed = lfs::core::Tensor::zeros(current_shape, param.device());

        // Copy data (manual memory copy as workaround)
        size_t bytes_to_copy = state.size * param.numel() / param.shape()[0] * sizeof(float);
        cudaMemcpy(exp_avg_trimmed.ptr<float>(), state.exp_avg.ptr<float>(), bytes_to_copy, cudaMemcpyDeviceToDevice);
        cudaMemcpy(exp_avg_sq_trimmed.ptr<float>(), state.exp_avg_sq.ptr<float>(), bytes_to_copy, cudaMemcpyDeviceToDevice);

        // Now concatenate
        std::vector<lfs::core::Tensor> exp_avg_parts = {exp_avg_trimmed, new_zeros};
        std::vector<lfs::core::Tensor> exp_avg_sq_parts = {exp_avg_sq_trimmed, new_zeros};

        state.exp_avg = lfs::core::Tensor::cat(exp_avg_parts, 0);
        state.exp_avg_sq = lfs::core::Tensor::cat(exp_avg_sq_parts, 0);

        // Update capacity and size (no pre-allocation, exact fit)
        state.capacity = new_size;
        state.size = new_size;

        LOG_DEBUG("Extended optimizer state for {} by {} parameters (size: {} -> {})",
                  name, n_new, state.size - n_new, state.size);
    }

    size_t AdamOptimizer::compute_new_capacity(size_t current_capacity, size_t required_size) const {
        size_t new_capacity;
        if (current_capacity == 0) {
            // First allocation: use initial_capacity if set, otherwise exact fit with some growth
            if (config_.initial_capacity > 0) {
                new_capacity = std::max(config_.initial_capacity, required_size);
                LOG_DEBUG("compute_new_capacity: initial allocation with config.initial_capacity={}, required_size={} -> new_capacity={}",
                          config_.initial_capacity, required_size, new_capacity);
            } else {
                // Default: allocate 150% of required to avoid immediate reallocation
                new_capacity = static_cast<size_t>(required_size * 1.5f);
                LOG_DEBUG("compute_new_capacity: initial allocation (no initial_capacity set), required_size={} -> new_capacity={} (1.5x)",
                          required_size, new_capacity);
            }
            return new_capacity;
        }

        // Grow by growth_factor (like std::vector uses 1.5x or 2x)
        size_t grown_capacity = static_cast<size_t>(current_capacity * config_.growth_factor);
        new_capacity = std::max(grown_capacity, required_size);
        LOG_DEBUG("compute_new_capacity: growth, current_capacity={}, required_size={}, growth_factor={} -> new_capacity={}",
                  current_capacity, required_size, config_.growth_factor, new_capacity);
        return new_capacity;
    }

    const AdamParamState* AdamOptimizer::get_state(ParamType type) const {
        auto name = param_name(type);
        auto it = states_.find(name);
        if (it == states_.end()) {
            return nullptr;
        }
        // NOTE: Returns the state with full capacity tensors
        // Caller should use state->size to know the actual used size
        // The exp_avg/exp_avg_sq tensors may have shape[0] > size due to pre-allocation
        return &it->second;
    }

    int64_t AdamOptimizer::get_step_count(ParamType type) const {
        auto name = param_name(type);
        auto it = states_.find(name);
        if (it == states_.end()) {
            return 0;
        }
        return it->second.step_count;
    }

    void AdamOptimizer::set_state(ParamType type, const AdamParamState& state) {
        auto name = param_name(type);
        states_[name] = state;
        LOG_DEBUG("Set optimizer state for {} (size: {}, capacity: {})",
                  name, state.size, state.capacity);
    }

    void AdamOptimizer::add_new_params(ParamType type, const lfs::core::Tensor& new_values, bool validate) {
        auto& param = get_param(type);
        auto& grad = get_grad(type);

        // Validation: check that new_values has compatible shape
        if (validate) {
            if (new_values.ndim() != param.ndim()) {
                throw std::runtime_error(
                    "add_new_params: new_values rank (" + std::to_string(new_values.ndim()) +
                    ") doesn't match existing parameter rank (" + std::to_string(param.ndim()) + ")"
                );
            }

            // Check that all dimensions except first match
            for (size_t i = 1; i < param.ndim(); i++) {
                if (new_values.shape()[i] != param.shape()[i]) {
                    throw std::runtime_error(
                        "add_new_params: new_values shape mismatch at dimension " + std::to_string(i)
                    );
                }
            }

            // Check device matches
            if (new_values.device() != param.device()) {
                throw std::runtime_error(
                    "add_new_params: new_values device doesn't match existing parameter device"
                );
            }
        }

        size_t n_new = new_values.shape()[0];
        size_t n_current = param.shape()[0];

        // OPTIMIZATION: Use tensor concatenation (requires allocation)
        // NOTE: For a zero-allocation version, SplatData would need to pre-allocate
        // with excess capacity and use slicing. This is a cleaner API but allocates.
        param = lfs::core::Tensor::cat(std::vector<lfs::core::Tensor>{param, new_values}, 0);

        // Verify param is still valid after cat
        if (!param.is_valid() || param.ndim() == 0) {
            throw std::runtime_error("add_new_params: parameter became invalid after cat()!");
        }

        // Re-obtain gradient reference after modifying param
        // (in case the reference became stale)
        auto& grad_updated = get_grad(type);

        // Verify grad is valid
        if (!grad_updated.is_valid() || grad_updated.ndim() == 0) {
            throw std::runtime_error("add_new_params: gradient is invalid or rank-0! param.ndim()=" +
                                   std::to_string(param.ndim()) + ", grad.ndim()=" + std::to_string(grad_updated.ndim()));
        }

        // Extend gradient with zeros
        std::vector<size_t> grad_dims(param.ndim());
        for (size_t i = 0; i < param.ndim(); i++) {
            grad_dims[i] = (i == 0) ? n_new : param.shape()[i];
        }
        auto zeros_grad = lfs::core::Tensor::zeros(lfs::core::TensorShape(grad_dims), param.device());

        LOG_DEBUG("  add_new_params gradient concatenation:");
        LOG_DEBUG("    existing grad: shape[0]={}, ndim={}", grad_updated.shape()[0], grad_updated.ndim());
        LOG_DEBUG("    zeros_grad: shape[0]={}, ndim={}", zeros_grad.shape()[0], zeros_grad.ndim());

        grad_updated = lfs::core::Tensor::cat(std::vector<lfs::core::Tensor>{grad_updated, zeros_grad}, 0);

        if (grad_updated.numel() == 0) {
            LOG_ERROR("  Gradient concatenation failed! Resulting tensor is empty");
        } else {
            LOG_DEBUG("    result grad: shape[0]={}, ndim={}", grad_updated.shape()[0], grad_updated.ndim());
        }

        // Extend optimizer state (this can be optimized with capacity tracking)
        extend_state_for_new_params(type, n_new);
    }

    void AdamOptimizer::add_new_params_gather(ParamType type, const lfs::core::Tensor& indices) {
        LOG_DEBUG("add_new_params_gather for {}", param_name(type));

        // Get parameter and gradient references
        auto& param = get_param(type);
        auto& grad = get_grad(type);

        if (!param.is_valid()) {
            LOG_ERROR("add_new_params_gather: parameter {} not initialized", param_name(type));
            return;
        }

        if (indices.device() != param.device()) {
            LOG_ERROR("add_new_params_gather: indices device doesn't match parameter device");
            return;
        }

        size_t n_new = indices.numel();
        size_t n_current = param.shape()[0];

        LOG_DEBUG("  Appending {} new values to {} existing values", n_new, n_current);

        // Use fused append_gather() operation - NO INTERMEDIATE ALLOCATION!
        param.append_gather(indices);

        // Extend gradient with zeros using append_zeros() - NO INTERMEDIATE ALLOCATION!
        LOG_DEBUG("  add_new_params_gather gradient extension:");
        LOG_DEBUG("    existing grad: shape[0]={}, capacity={}, ndim={}",
                  grad.shape()[0], grad.capacity(), grad.ndim());

        grad.append_zeros(n_new);

        LOG_DEBUG("    result grad: shape[0]={}, capacity={}, ndim={}",
                  grad.shape()[0], grad.capacity(), grad.ndim());

        // Extend optimizer state (this can be optimized with capacity tracking)
        extend_state_for_new_params(type, n_new);
    }

    void AdamOptimizer::relocate_params_at_indices(ParamType type, const std::vector<int64_t>& indices) {
        if (indices.empty()) return;

        auto& param = get_param(type);

        // Validation: check indices are in bounds
        for (auto idx : indices) {
            if (idx < 0 || static_cast<size_t>(idx) >= param.shape()[0]) {
                throw std::runtime_error(
                    "relocate_params_at_indices: index " + std::to_string(idx) +
                    " out of bounds [0, " + std::to_string(param.shape()[0]) + ")"
                );
            }
        }

        // Copy indices to GPU once, then use fast GPU version
        int64_t* indices_device_ptr;
        cudaMalloc(&indices_device_ptr, indices.size() * sizeof(int64_t));
        cudaMemcpy(indices_device_ptr, indices.data(), indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

        relocate_params_at_indices_gpu(type, indices_device_ptr, indices.size());

        cudaFree(indices_device_ptr);
    }

    void AdamOptimizer::relocate_params_at_indices_gpu(ParamType type, const int64_t* indices_device, size_t n_indices) {
        if (n_indices == 0) return;

        auto& param = get_param(type);
        auto& grad = get_grad(type);
        auto name = param_name(type);

        // Calculate row size for gradients
        auto grad_shape = grad.shape();
        int grad_row_size = 1;
        for (size_t i = 1; i < grad_shape.rank(); i++) {
            grad_row_size *= grad_shape[i];
        }

        // Zero out gradients using batched GPU kernel (FAST!)
        fast_lfs::optimizer::zero_rows_at_indices(
            grad.template ptr<float>(),
            indices_device,
            n_indices,
            grad_row_size
        );

        // Ensure optimizer state exists
        if (states_.find(name) == states_.end()) {
            LOG_DEBUG("State for {} not initialized yet, skipping reset", name);
            return;
        }

        auto& state = states_[name];

        // Calculate row size for optimizer state
        auto state_shape = state.exp_avg.shape();
        int state_row_size = 1;
        for (size_t i = 1; i < state_shape.rank(); i++) {
            state_row_size *= state_shape[i];
        }

        // Zero out optimizer state using batched GPU kernel (FAST!)
        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg.template ptr<float>(),
            indices_device,
            n_indices,
            state_row_size
        );

        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg_sq.template ptr<float>(),
            indices_device,
            n_indices,
            state_row_size
        );

        LOG_DEBUG("relocate_params_at_indices_gpu: Reset state and gradients for {} at {} indices (batched GPU kernel)",
                  name, n_indices);
    }

} // namespace lfs::training
