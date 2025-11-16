/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/splat_data.hpp"
#include <array>
#include <string>
#include <unordered_map>

/**
 * LibTorch-free Adam Optimizer for Gaussian Splatting
 *
 * This optimizer provides a safe, efficient implementation of the Adam algorithm
 * with special support for MCMC-based training strategies (parameter addition/relocation).
 *
 * ===== USAGE EXAMPLE - SAFE API (RECOMMENDED) =====
 *
 * ```cpp
 * // Setup
 * lfs::core::SplatData splat_data = load_initial_gaussians();
 * splat_data.allocate_gradients();
 *
 * AdamConfig config;
 * config.lr = 1e-3f;
 * AdamOptimizer optimizer(splat_data, config);
 *
 * // Regular training loop
 * for (int iter = 0; iter < num_iterations; iter++) {
 *     compute_gradients(splat_data);  // Your backward pass
 *     optimizer.step(iter);
 *     optimizer.zero_grad(iter);
 *
 *     // MCMC: Add new Gaussians (atomic - safe!)
 *     if (should_add_gaussians(iter)) {
 *         auto new_gaussians = generate_new_gaussians();
 *         optimizer.add_new_params(ParamType::Means, new_gaussians);
 *         // Parameters, gradients, and optimizer state are ALL updated atomically!
 *         // No need to manually update gradients or call extend_state_for_new_params()
 *     }
 *
 *     // MCMC: Relocate dead Gaussians (atomic - safe!)
 *     if (should_relocate(iter)) {
 *         auto dead_indices = find_dead_gaussians();
 *         optimizer.relocate_params_at_indices(ParamType::Means, dead_indices);
 *         // Gradients are zeroed AND optimizer state is reset - all in one call!
 *     }
 * }
 * ```
 *
 * ===== UNSAFE PATTERN (AVOID - ERROR PRONE) =====
 *
 * ```cpp
 * // DON'T DO THIS - manual synchronization is error-prone:
 * auto new_means = generate_new_gaussians();
 * splat_data.means() = Tensor::cat({splat_data.means(), new_means}, 0);  // Update params
 * splat_data.means_grad() = Tensor::cat({...}, 0);  // Must remember to update grads!
 * optimizer.extend_state_for_new_params(ParamType::Means, n_new);  // Must remember this too!
 * // If you forget any step above, you get silent bugs or crashes!
 * ```
 *
 * ===== KEY BENEFITS OF SAFE API =====
 *
 * 1. **Atomic Operations**: Parameters, gradients, and optimizer state are always synchronized
 * 2. **Validation**: Automatic shape/device checking prevents silent bugs
 * 3. **Cleaner Code**: No manual tensor concatenation or state management
 * 4. **Safer MCMC**: Can't forget to zero gradients when relocating parameters
 *
 * ===== PERFORMANCE OPTIMIZATIONS =====
 *
 * The optimizer uses **capacity-based growth** (like std::vector) to minimize GPU allocations:
 *
 * - **Optimizer state** (exp_avg, exp_avg_sq) pre-allocates extra capacity
 * - When adding parameters, if capacity is available, NO allocation occurs (fast path!)
 * - Growth factor of 1.5x means ~50% fewer reallocations vs exact-fit allocation
 * - For known workloads, set `initial_capacity` to max Gaussians to avoid ALL reallocations
 *
 * ```cpp
 * AdamConfig config;
 * config.lr = 1e-3f;
 * config.initial_capacity = 1'000'000;  // Pre-allocate for 1M Gaussians
 * config.growth_factor = 1.5f;           // Grow by 50% when capacity exceeded
 * AdamOptimizer optimizer(splat_data, config);
 * // First 1M Gaussians: ZERO allocations for optimizer state!
 * ```
 *
 * **Note**: Parameters/gradients in SplatData still use concatenation (requires allocation).
 * For zero-copy parameter updates, SplatData would need similar capacity tracking.
 *
 * ===== LOW-LEVEL API (LEGACY) =====
 *
 * The low-level methods (reset_state_at_indices, extend_state_for_new_params) are still
 * available for backwards compatibility and testing, but the safe API is recommended.
 */

namespace lfs::training {

    struct AdamConfig {
        float lr = 1e-3f;  // Default learning rate (used if per-param LRs not set)
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;

        // Per-parameter learning rates (optional - if not set, uses lr)
        std::unordered_map<std::string, float> param_lrs;

        // Memory optimization settings (for optimizer state, not parameters)
        float growth_factor = 1.5f;      // Multiply capacity by this when growing (like std::vector)
                                         // Default 1.5x means ~50% fewer reallocations than exact-fit
        size_t initial_capacity = 0;     // Initial capacity (0 = auto, >0 = pre-allocate to this size)
                                         // Set to max expected Gaussians to avoid ALL reallocations
    };

    struct AdamParamState {
        lfs::core::Tensor exp_avg;       // First moment estimate
        lfs::core::Tensor exp_avg_sq;    // Second moment estimate
        int64_t step_count = 0;

        // Capacity tracking for efficient growth
        size_t capacity = 0;             // Total allocated capacity (first dimension)
        size_t size = 0;                 // Currently used size (first dimension)
    };

    // Parameter type enum (public for state manipulation)
    enum class ParamType {
        Means,
        Sh0,
        ShN,
        Scaling,
        Rotation,
        Opacity
    };

    class AdamOptimizer {
    public:
        explicit AdamOptimizer(lfs::core::SplatData& splat_data, const AdamConfig& config);

        // Main optimization step
        void step(int iteration);

        // Zero gradients
        void zero_grad(int iteration);

        // Update learning rate (global or per-parameter)
        void set_lr(float lr) { config_.lr = lr; }
        float get_lr() const { return config_.lr; }

        // Set per-parameter learning rate
        void set_param_lr(ParamType type, float lr) {
            config_.param_lrs[param_name(type)] = lr;
        }

        // Get per-parameter learning rate (falls back to global lr if not set)
        float get_param_lr(ParamType type) const {
            auto name = param_name(type);
            auto it = config_.param_lrs.find(name);
            if (it != config_.param_lrs.end()) {
                return it->second;
            }
            return config_.lr;
        }

        // Check if per-parameter learning rate is explicitly set
        bool has_param_lr(ParamType type) const {
            auto name = param_name(type);
            return config_.param_lrs.find(name) != config_.param_lrs.end();
        }

        // Get all parameter types (for scheduler to iterate)
        static constexpr std::array<ParamType, 6> all_param_types() {
            return {ParamType::Means, ParamType::Sh0, ParamType::ShN,
                    ParamType::Scaling, ParamType::Rotation, ParamType::Opacity};
        }

        // ===== SAFE MCMC OPERATIONS =====
        // These methods atomically update both parameters and optimizer state

        /**
         * Add new parameters (e.g., new Gaussians from MCMC split/clone)
         * Atomically extends parameters, gradients, and optimizer state.
         *
         * @param type Parameter type to extend
         * @param new_values New parameter values to append
         * @param validate If true, checks that new_values shape matches existing (default: true)
         */
        void add_new_params(ParamType type, const lfs::core::Tensor& new_values, bool validate = false);

        /**
         * Add new parameters using fused append_gather() operation.
         * This is more efficient than add_new_params() as it avoids allocating
         * intermediate tensors from index_select().
         *
         * Requirements:
         * - Parameter must have pre-allocated capacity (via reserve())
         * - indices must be on the same device as the parameter
         *
         * @param type Parameter type to extend
         * @param indices Indices to gather from existing parameter values
         */
        void add_new_params_gather(ParamType type, const lfs::core::Tensor& indices);

        /**
         * Reset optimizer state at specific indices (e.g., relocated dead Gaussians)
         * Also zeros out the corresponding gradients to ensure clean state.
         *
         * @param type Parameter type
         * @param indices Indices to reset (CPU vector - will be copied to GPU)
         */
        void relocate_params_at_indices(ParamType type, const std::vector<int64_t>& indices);

        /**
         * Reset optimizer state at specific indices (FAST VERSION - indices already on GPU)
         * Also zeros out the corresponding gradients to ensure clean state.
         *
         * @param type Parameter type
         * @param indices_device GPU pointer to indices (must be valid device pointer!)
         * @param n_indices Number of indices
         */
        void relocate_params_at_indices_gpu(ParamType type, const int64_t* indices_device, size_t n_indices);

        // ===== LOW-LEVEL STATE MANIPULATION =====
        // These are kept for backwards compatibility and testing, but prefer the safe methods above

        // Reset optimizer state at indices (does NOT zero gradients - use relocate_params_at_indices instead)
        void reset_state_at_indices(ParamType type, const std::vector<int64_t>& indices);

        // Extend optimizer state (does NOT update parameters - use add_new_params instead)
        void extend_state_for_new_params(ParamType type, size_t n_new);

        // Access to state for testing
        const AdamParamState* get_state(ParamType type) const;
        int64_t get_step_count(ParamType type) const;

        // Set optimizer state (for manual state updates in strategy operations)
        void set_state(ParamType type, const AdamParamState& state);

    private:
        AdamConfig config_;
        lfs::core::SplatData& splat_data_;

        // Optimizer state for each parameter
        std::unordered_map<std::string, AdamParamState> states_;

        // Get param and grad by type
        lfs::core::Tensor& get_param(ParamType type);
        lfs::core::Tensor& get_grad(ParamType type);
        std::string param_name(ParamType type) const;

        // Initialize state for a parameter
        void init_state(ParamType type);

        // Step for a single parameter
        void step_param(ParamType type, int iteration);

        // Capacity management helpers
        void ensure_param_capacity(ParamType type, size_t required_size);
        void ensure_state_capacity(ParamType type, size_t required_size);
        size_t compute_new_capacity(size_t current_capacity, size_t required_size) const;
    };

} // namespace lfs::training
