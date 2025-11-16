/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcmc_kernels.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/adjacent_difference.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

namespace lfs::training::mcmc {

    // Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
    __global__ void relocation_kernel(
        const float* opacities,
        const float* scales,
        const int32_t* ratios,
        const float* binoms,
        int n_max,
        float* new_opacities,
        float* new_scales,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        int n_idx = ratios[idx];
        float denom_sum = 0.0f;

        // Compute new opacity: 1 - (1 - old_opacity)^(1/n_idx)
        // Match legacy gsplat implementation exactly - no clamping, no safety checks
        // Use pow() instead of powf() to match legacy exactly
        new_opacities[idx] = 1.0f - pow(1.0f - opacities[idx], 1.0f / n_idx);

        // Compute new scale
        for (int i = 1; i <= n_idx; ++i) {
            for (int k = 0; k <= (i - 1); ++k) {
                float bin_coeff = binoms[(i - 1) * n_max + k];
                float term = (pow(-1.0f, k) / sqrt(static_cast<float>(k + 1))) *
                             pow(new_opacities[idx], k + 1);
                denom_sum += (bin_coeff * term);
            }
        }

        // Match legacy exactly - use raw opacity, no division by zero check
        float coeff = (opacities[idx] / denom_sum);
        for (int i = 0; i < 3; ++i) {
            new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i];
        }
    }

    void launch_relocation_kernel(
        const float* opacities,
        const float* scales,
        const int32_t* ratios,
        const float* binoms,
        int n_max,
        float* new_opacities,
        float* new_scales,
        size_t N,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        relocation_kernel<<<grid, threads, 0, cuda_stream>>>(
            opacities,
            scales,
            ratios,
            binoms,
            n_max,
            new_opacities,
            new_scales,
            N);
    }

    // Helper: Convert raw quaternion to rotation matrix
    __device__ inline void raw_quat_to_rotmat(const float* raw_quat, float* R) {
        float w = raw_quat[0], x = raw_quat[1], y = raw_quat[2], z = raw_quat[3];

        // Normalize
        float inv_norm = fminf(rsqrtf(x * x + y * y + z * z + w * w), 1e+12f);
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
        w *= inv_norm;

        float x2 = x * x, y2 = y * y, z2 = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float wx = w * x, wy = w * y, wz = w * z;

        // Column-major order (matching glm)
        R[0] = 1.f - 2.f * (y2 + z2);
        R[1] = 2.f * (xy + wz);
        R[2] = 2.f * (xz - wy);

        R[3] = 2.f * (xy - wz);
        R[4] = 1.f - 2.f * (x2 + z2);
        R[5] = 2.f * (yz + wx);

        R[6] = 2.f * (xz + wy);
        R[7] = 2.f * (yz - wx);
        R[8] = 1.f - 2.f * (x2 + y2);
    }

    // Helper: Matrix-vector multiplication (3x3 * 3x1)
    __device__ inline void matvec3(const float* M, const float* v, float* result) {
        result[0] = M[0] * v[0] + M[3] * v[1] + M[6] * v[2];
        result[1] = M[1] * v[0] + M[4] * v[1] + M[7] * v[2];
        result[2] = M[2] * v[0] + M[5] * v[1] + M[8] * v[2];
    }

    __global__ void add_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        const float* noise,
        float* means,
        float current_lr,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        size_t idx_3d = 3 * idx;
        size_t idx_4d = 4 * idx;

        // Compute S^2 (diagonal matrix from exp(2 * raw_scale))
        float S2[9] = {0};
        S2[0] = __expf(2.f * raw_scales[idx_3d + 0]);
        S2[4] = __expf(2.f * raw_scales[idx_3d + 1]);
        S2[8] = __expf(2.f * raw_scales[idx_3d + 2]);

        // Get rotation matrix R from quaternion
        float R[9];
        raw_quat_to_rotmat(raw_quats + idx_4d, R);

        // Compute R * S^2 (temp storage)
        float RS2[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                RS2[i * 3 + j] = R[i * 3 + 0] * S2[0 * 3 + j] +
                                 R[i * 3 + 1] * S2[1 * 3 + j] +
                                 R[i * 3 + 2] * S2[2 * 3 + j];
            }
        }

        // Compute covariance = R * S^2 * R^T
        float covariance[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                covariance[i * 3 + j] = RS2[i * 3 + 0] * R[j * 3 + 0] +
                                        RS2[i * 3 + 1] * R[j * 3 + 1] +
                                        RS2[i * 3 + 2] * R[j * 3 + 2];
            }
        }

        // Transform noise: transformed_noise = covariance * noise
        float transformed_noise[3];
        float noise_vec[3] = {noise[idx_3d], noise[idx_3d + 1], noise[idx_3d + 2]};
        matvec3(covariance, noise_vec, transformed_noise);

        // Compute opacity-based scaling factor
        float opacity = __frcp_rn(1.f + __expf(-raw_opacities[idx]));  // sigmoid
        float op_sigmoid = __frcp_rn(1.f + __expf(100.f * opacity - 0.5f));
        float noise_factor = current_lr * op_sigmoid;

        // Add scaled noise to means
        means[idx_3d + 0] += noise_factor * transformed_noise[0];
        means[idx_3d + 1] += noise_factor * transformed_noise[1];
        means[idx_3d + 2] += noise_factor * transformed_noise[2];
    }

    void launch_add_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        const float* noise,
        float* means,
        float current_lr,
        size_t N,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        add_noise_kernel<<<grid, threads, 0, cuda_stream>>>(
            raw_opacities,
            raw_scales,
            raw_quats,
            noise,
            means,
            current_lr,
            N);
    }

    // Fused gather kernel - collects all parameters at once
    __global__ void gather_gaussian_params_kernel(
        const int64_t* __restrict__ indices,
        const float* __restrict__ src_means,
        const float* __restrict__ src_sh0,
        const float* __restrict__ src_shN,
        const float* __restrict__ src_scales,
        const float* __restrict__ src_rotations,
        const float* __restrict__ src_opacities,
        float* __restrict__ dst_means,
        float* __restrict__ dst_sh0,
        float* __restrict__ dst_shN,
        float* __restrict__ dst_scales,
        float* __restrict__ dst_rotations,
        float* __restrict__ dst_opacities,
        size_t n_samples,
        size_t sh_rest,
        int opacity_dim,
        size_t N) {  // Add N parameter for bounds checking

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        int64_t src_idx = indices[idx];

        // Bounds check - CRITICAL for safety
        if (src_idx < 0 || src_idx >= static_cast<int64_t>(N)) {
            // Invalid index - skip this gather (leave output uninitialized or zero it)
            return;
        }

        // Gather means [3]
        for (int i = 0; i < 3; ++i) {
            dst_means[idx * 3 + i] = src_means[src_idx * 3 + i];
        }

        // Gather sh0 [N, 1, 3] -> output [n_samples, 1, 3]
        // Memory layout: each Gaussian has 1*3 = 3 floats
        for (int i = 0; i < 3; ++i) {
            dst_sh0[idx * 3 + i] = src_sh0[src_idx * 3 + i];
        }

        // Gather shN [sh_rest, 3]
        for (size_t i = 0; i < sh_rest * 3; ++i) {
            dst_shN[idx * sh_rest * 3 + i] = src_shN[src_idx * sh_rest * 3 + i];
        }

        // Gather scales [3]
        for (int i = 0; i < 3; ++i) {
            dst_scales[idx * 3 + i] = src_scales[src_idx * 3 + i];
        }

        // Gather rotations [4]
        for (int i = 0; i < 4; ++i) {
            dst_rotations[idx * 4 + i] = src_rotations[src_idx * 4 + i];
        }

        // Gather opacities [1] or []
        if (opacity_dim == 1) {
            dst_opacities[idx] = src_opacities[src_idx];
        } else {
            dst_opacities[idx] = src_opacities[src_idx];
        }
    }

    void launch_gather_gaussian_params(
        const int64_t* indices,
        const float* src_means,
        const float* src_sh0,
        const float* src_shN,
        const float* src_scales,
        const float* src_rotations,
        const float* src_opacities,
        float* dst_means,
        float* dst_sh0,
        float* dst_shN,
        float* dst_scales,
        float* dst_rotations,
        float* dst_opacities,
        size_t n_samples,
        size_t sh_rest,
        int opacity_dim,
        size_t N,  // Add N parameter
        void* stream) {

        if (n_samples == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((n_samples + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        gather_gaussian_params_kernel<<<grid, threads, 0, cuda_stream>>>(
            indices,
            src_means,
            src_sh0,
            src_shN,
            src_scales,
            src_rotations,
            src_opacities,
            dst_means,
            dst_sh0,
            dst_shN,
            dst_scales,
            dst_rotations,
            dst_opacities,
            n_samples,
            sh_rest,
            opacity_dim,
            N);
    }

    // Kernel to update scaling and opacity at specific indices (avoids index_put_ which loses capacity)
    __global__ void update_scaling_opacity_kernel(
        const int64_t* __restrict__ indices,
        const float* __restrict__ new_scaling,      // [n_indices, 3]
        const float* __restrict__ new_opacity_raw,  // [n_indices] or [n_indices, 1]
        float* __restrict__ scaling_raw,
        float* __restrict__ opacity_raw,
        size_t n_indices,
        int opacity_dim,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_indices)
            return;

        int64_t target_idx = indices[idx];

        // Bounds check
        if (target_idx < 0 || target_idx >= static_cast<int64_t>(N)) {
            return;
        }

        // Update scaling [3]
        for (int i = 0; i < 3; ++i) {
            scaling_raw[target_idx * 3 + i] = new_scaling[idx * 3 + i];
        }

        // Update opacity
        opacity_raw[target_idx] = new_opacity_raw[idx];
    }

    void launch_update_scaling_opacity(
        const int64_t* indices,
        const float* new_scaling,
        const float* new_opacity_raw,
        float* scaling_raw,
        float* opacity_raw,
        size_t n_indices,
        int opacity_dim,
        size_t N,
        void* stream) {

        if (n_indices == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((n_indices + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        update_scaling_opacity_kernel<<<grid, threads, 0, cuda_stream>>>(
            indices,
            new_scaling,
            new_opacity_raw,
            scaling_raw,
            opacity_raw,
            n_indices,
            opacity_dim,
            N);
    }

    // Fused copy kernel - copies all parameters from src_indices to dst_indices
    __global__ void copy_gaussian_params_kernel(
        const int64_t* __restrict__ src_indices,
        const int64_t* __restrict__ dst_indices,
        float* __restrict__ means,
        float* __restrict__ sh0,
        float* __restrict__ shN,
        float* __restrict__ scales,
        float* __restrict__ rotations,
        float* __restrict__ opacities,
        size_t n_copy,
        size_t sh_rest,
        int opacity_dim,
        size_t N) {  // Add N parameter for bounds checking

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_copy)
            return;

        int64_t src_idx = src_indices[idx];
        int64_t dst_idx = dst_indices[idx];

        // Bounds check - CRITICAL for safety
        if (src_idx < 0 || src_idx >= static_cast<int64_t>(N) ||
            dst_idx < 0 || dst_idx >= static_cast<int64_t>(N)) {
            // Invalid index - skip this copy
            return;
        }

        // Copy means [3]
        for (int i = 0; i < 3; ++i) {
            means[dst_idx * 3 + i] = means[src_idx * 3 + i];
        }

        // Copy sh0 [1, 3] -> [3]
        for (int i = 0; i < 3; ++i) {
            sh0[dst_idx * 3 + i] = sh0[src_idx * 3 + i];
        }

        // Copy shN [sh_rest, 3]
        for (size_t i = 0; i < sh_rest * 3; ++i) {
            shN[dst_idx * sh_rest * 3 + i] = shN[src_idx * sh_rest * 3 + i];
        }

        // Copy scales [3]
        for (int i = 0; i < 3; ++i) {
            scales[dst_idx * 3 + i] = scales[src_idx * 3 + i];
        }

        // Copy rotations [4]
        for (int i = 0; i < 4; ++i) {
            rotations[dst_idx * 4 + i] = rotations[src_idx * 4 + i];
        }

        // Copy opacities [1] or []
        if (opacity_dim == 1) {
            opacities[dst_idx] = opacities[src_idx];
        } else {
            opacities[dst_idx] = opacities[src_idx];
        }
    }

    void launch_copy_gaussian_params(
        const int64_t* src_indices,
        const int64_t* dst_indices,
        float* means,
        float* sh0,
        float* shN,
        float* scales,
        float* rotations,
        float* opacities,
        size_t n_copy,
        size_t sh_rest,
        int opacity_dim,
        size_t N,  // Add N parameter
        void* stream) {

        if (n_copy == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((n_copy + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        copy_gaussian_params_kernel<<<grid, threads, 0, cuda_stream>>>(
            src_indices,
            dst_indices,
            means,
            sh0,
            shN,
            scales,
            rotations,
            opacities,
            n_copy,
            sh_rest,
            opacity_dim,
            N);
    }

    // Histogram kernel using atomics - counts occurrences of each index
    __global__ void histogram_kernel(
        const int64_t* __restrict__ indices,
        int32_t* __restrict__ counts,
        size_t n_samples,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        int64_t index = indices[idx];

        // Bounds check
        if (index < 0 || index >= static_cast<int64_t>(N))
            return;

        // Atomic increment
        atomicAdd(&counts[index], 1);
    }

    void launch_histogram(
        const int64_t* indices,
        int32_t* counts,
        size_t n_samples,
        size_t N,
        void* stream) {

        if (n_samples == 0)
            return;

        dim3 threads(256);
        dim3 grid((n_samples + threads.x - 1) / threads.x);
        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        histogram_kernel<<<grid, threads, 0, cuda_stream>>>(
            indices, counts, n_samples, N);
    }

    // Smarter histogram: Use hash map-style approach with sorting
    // This works well when n_samples << N (which is our case)
    __global__ void histogram_gather_sorted_kernel(
        const int64_t* __restrict__ indices,
        int32_t* __restrict__ output_counts,
        size_t n_samples,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        int64_t my_index = indices[idx];

        // Bounds check
        if (my_index < 0 || my_index >= static_cast<int64_t>(N)) {
            output_counts[idx] = 0;
            return;
        }

        // Count occurrences: scan forward until we find a different index
        // This works ONLY if indices are sorted or if we accept O(n) per thread
        // Since indices are NOT sorted, we do linear scan (unavoidable without large temp storage)

        // Optimization: Use warp-level primitives to speed up counting
        int32_t count = 0;

        // Each thread scans the array looking for matches
        // This is O(n) per thread, total O(n*n_samples)
        // BUT: We can optimize using warp primitives

        const int WARP_SIZE = 32;
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        // Each warp cooperatively counts for one index
        for (size_t i = lane_id; i < n_samples; i += WARP_SIZE) {
            if (indices[i] == my_index) {
                count++;
            }
        }

        // Warp-level reduction to sum counts
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            count += __shfl_down_sync(0xffffffff, count, offset);
        }

        // First lane writes the result
        if (lane_id == 0) {
            output_counts[idx] = count;
        }
    }

    void launch_histogram_sort(
        const int64_t* indices,
        int32_t* output_counts,
        size_t n_samples,
        void* stream) {

        if (n_samples == 0)
            return;

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        // Algorithm: Sort indices, then use adjacent_difference to find run boundaries,
        // then use inclusive_scan to count run lengths, then scatter back to original positions

        // Step 1: Create position array and copy indices
        thrust::device_vector<int32_t> orig_positions(n_samples);
        thrust::sequence(thrust::cuda::par.on(cuda_stream), orig_positions.begin(), orig_positions.end());

        thrust::device_vector<int64_t> sorted_indices(indices, indices + n_samples);

        // Step 2: Sort indices while tracking original positions
        thrust::sort_by_key(thrust::cuda::par.on(cuda_stream),
                           sorted_indices.begin(), sorted_indices.end(),
                           orig_positions.begin());

        // Step 3: Mark segment boundaries (1 where index changes, 0 otherwise)
        thrust::device_vector<int32_t> head_flags(n_samples);
        thrust::adjacent_difference(thrust::cuda::par.on(cuda_stream),
                                   sorted_indices.begin(), sorted_indices.end(),
                                   head_flags.begin(),
                                   thrust::not_equal_to<int64_t>());
        // First element is always a segment head
        if (n_samples > 0) {
            thrust::fill_n(thrust::cuda::par.on(cuda_stream), head_flags.begin(), 1, 1);
        }

        // Step 4: Compute run lengths with exclusive_scan_by_key
        // This gives each element its position within its segment
        thrust::device_vector<int32_t> run_positions(n_samples);
        thrust::device_vector<int32_t> ones(n_samples, 1);

        thrust::exclusive_scan_by_key(thrust::cuda::par.on(cuda_stream),
                                     sorted_indices.begin(), sorted_indices.end(),
                                     ones.begin(),
                                     run_positions.begin());

        // Step 5: Find the tail of each segment and compute run length
        // Use a kernel to compute the count for each element
        thrust::device_vector<int32_t> run_counts(n_samples);

        thrust::transform(thrust::cuda::par.on(cuda_stream),
                         thrust::make_counting_iterator<int>(0),
                         thrust::make_counting_iterator<int>(n_samples),
                         run_counts.begin(),
                         [sorted_indices_ptr = thrust::raw_pointer_cast(sorted_indices.data()),
                          run_positions_ptr = thrust::raw_pointer_cast(run_positions.data()),
                          n_samples] __device__ (int idx) {
                              int64_t my_index = sorted_indices_ptr[idx];
                              int my_pos = run_positions_ptr[idx];

                              // Find the last occurrence of this index
                              int count = 1;
                              if (idx + 1 < n_samples && sorted_indices_ptr[idx + 1] == my_index) {
                                  // Not the last in segment, find it
                                  for (int i = idx + 1; i < n_samples && sorted_indices_ptr[i] == my_index; ++i) {
                                      count = run_positions_ptr[i] + 1;
                                  }
                              } else {
                                  // Last in segment
                                  count = my_pos + 1;
                              }
                              return count;
                          });

        // Step 6: Scatter counts back to original positions
        thrust::scatter(thrust::cuda::par.on(cuda_stream),
                       run_counts.begin(), run_counts.end(),
                       orig_positions.begin(),
                       output_counts);
    }

    // Fused gather kernel for 2 tensors - replaces 2x index_select
    // OPTIMIZED: Unroll loops for common cases
    __global__ void gather_2tensors_kernel(
        const int64_t* __restrict__ indices,
        const float* __restrict__ src_a,
        const float* __restrict__ src_b,
        float* __restrict__ dst_a,
        float* __restrict__ dst_b,
        size_t n_samples,
        size_t dim_a,
        size_t dim_b,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        int64_t src_idx = indices[idx];

        // Bounds check - CRITICAL for safety
        if (src_idx < 0 || src_idx >= static_cast<int64_t>(N)) {
            // Zero the output for invalid indices
            for (size_t i = 0; i < dim_a; ++i) dst_a[idx * dim_a + i] = 0.0f;
            for (size_t i = 0; i < dim_b; ++i) dst_b[idx * dim_b + i] = 0.0f;
            return;
        }

        // Fast path for common case: dim_a=1, dim_b=3
        if (dim_a == 1 && dim_b == 3) {
            dst_a[idx] = src_a[src_idx];
            dst_b[idx * 3 + 0] = src_b[src_idx * 3 + 0];
            dst_b[idx * 3 + 1] = src_b[src_idx * 3 + 1];
            dst_b[idx * 3 + 2] = src_b[src_idx * 3 + 2];
            return;
        }

        // General case: Gather first tensor (dim_a elements)
        for (size_t i = 0; i < dim_a; ++i) {
            dst_a[idx * dim_a + i] = src_a[src_idx * dim_a + i];
        }

        // Gather second tensor (dim_b elements)
        for (size_t i = 0; i < dim_b; ++i) {
            dst_b[idx * dim_b + i] = src_b[src_idx * dim_b + i];
        }
    }

    void launch_gather_2tensors(
        const int64_t* indices,
        const float* src_a,
        const float* src_b,
        float* dst_a,
        float* dst_b,
        size_t n_samples,
        size_t dim_a,
        size_t dim_b,
        size_t N,
        void* stream) {

        if (n_samples == 0)
            return;

        dim3 threads(256);
        dim3 grid((n_samples + threads.x - 1) / threads.x);
        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        gather_2tensors_kernel<<<grid, threads, 0, cuda_stream>>>(
            indices,
            src_a,
            src_b,
            dst_a,
            dst_b,
            n_samples,
            dim_a,
            dim_b,
            N);
    }

} // namespace lfs::training::mcmc
