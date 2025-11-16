/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cstdio>
#include <string>
#include <torch/torch.h>
#include <tuple>

// Low-level CUDA kernel wrappers (used by autograd and manual implementations)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor& img1,
    torch::Tensor& img2,
    bool train);

torch::Tensor
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor& img1,
    torch::Tensor& img2,
    torch::Tensor& dL_dmap,
    torch::Tensor& dm_dmu1,
    torch::Tensor& dm_dsigma1_sq,
    torch::Tensor& dm_dsigma12);

// Context for manual SSIM forward/backward (like RasterizeContext)
struct SSIMContext {
    torch::Tensor img1;
    torch::Tensor img2;
    torch::Tensor dm_dmu1;
    torch::Tensor dm_dsigma1_sq;
    torch::Tensor dm_dsigma12;
    int64_t original_h;
    int64_t original_w;
    bool apply_valid_padding;
};

// Manual SSIM forward (no autograd) - returns (loss_value, context)
std::pair<float, SSIMContext> ssim_forward(
    const torch::Tensor& img1,
    const torch::Tensor& img2,
    bool apply_valid_padding = true);

// Manual SSIM backward (no autograd) - computes gradient w.r.t. img1
torch::Tensor ssim_backward(
    const SSIMContext& ctx,
    float grad_loss);  // Gradient of loss w.r.t. SSIM value (scalar)
