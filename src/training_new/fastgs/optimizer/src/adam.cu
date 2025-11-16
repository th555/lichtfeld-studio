/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam.h"
#include "optimizer_config.h"
#include "utils.h"

// Forward declare the kernels (defined in adam_api.cu)
namespace fast_lfs::optimizer::kernels::adam {
    __global__ void adam_step_vectorized_cu(
        float* param,
        float* exp_avg,
        float* exp_avg_sq,
        const float* param_grad,
        const int n_elements,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp);
}

void fast_lfs::optimizer::adam_step(
    float* param,
    float* exp_avg,
    float* exp_avg_sq,
    const float* param_grad,
    const int n_elements,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction1_rcp,
    const float bias_correction2_sqrt_rcp) {

    // Each thread processes 4 elements, so divide by 4
    const int n_threads = div_round_up(n_elements, 4);

    kernels::adam::adam_step_vectorized_cu<<<div_round_up(n_threads, config::block_size_adam_step), config::block_size_adam_step>>>(
        param,
        exp_avg,
        exp_avg_sq,
        param_grad,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1_rcp,
        bias_correction2_sqrt_rcp);
    CHECK_CUDA(config::debug, "adam step vectorized")
}
