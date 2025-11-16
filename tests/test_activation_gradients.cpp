/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>

// Test manual chain rule gradients vs autograd for activation functions
class ActivationGradientsTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_points = 100;
    }

    int n_points;
};

// Test sigmoid activation gradient
TEST_F(ActivationGradientsTest, SigmoidGradient) {
    const float tolerance = 1e-6f;

    // Create raw parameter
    auto opacity_raw = torch::randn({n_points, 1}, torch::kFloat32).cuda();

    // PATH 1: Autograd
    auto opacity_raw_auto = opacity_raw.clone();
    opacity_raw_auto.set_requires_grad(true);

    auto activated_auto = torch::sigmoid(opacity_raw_auto).squeeze(-1);

    // Simulate gradient from gsplat backward (random for testing)
    auto grad_activated = torch::randn_like(activated_auto);

    // Use autograd to get gradient on raw parameter
    activated_auto.backward(grad_activated);
    auto grad_raw_auto = opacity_raw_auto.grad();

    // PATH 2: Manual chain rule
    // Forward: activated = sigmoid(raw)
    // Backward: grad_raw = grad_activated * sigmoid'(raw)
    //         = grad_activated * sigmoid(raw) * (1 - sigmoid(raw))
    auto activated_manual = torch::sigmoid(opacity_raw).squeeze(-1);
    auto sigmoid_val = torch::sigmoid(opacity_raw);  // [n_points, 1]
    auto sigmoid_derivative = sigmoid_val * (1.0f - sigmoid_val);  // [n_points, 1]

    // grad_activated is [n_points], need to unsqueeze to [n_points, 1] for broadcasting
    auto grad_raw_manual = grad_activated.unsqueeze(-1) * sigmoid_derivative;

    // Compare
    auto diff = (grad_raw_auto - grad_raw_manual).abs().max().item().toFloat();
    auto scale = grad_raw_auto.abs().max().item().toFloat();

    EXPECT_LT(diff / (scale + 1e-8f), tolerance)
        << "Sigmoid gradient mismatch: max_diff=" << diff << ", scale=" << scale;

    std::cout << "Sigmoid: max_diff=" << diff << ", rel_err=" << (diff / (scale + 1e-8f)) << std::endl;
}

// Test exp activation gradient
TEST_F(ActivationGradientsTest, ExpGradient) {
    const float tolerance = 1e-6f;

    auto scaling_raw = torch::randn({n_points, 3}, torch::kFloat32).cuda();

    // PATH 1: Autograd
    auto scaling_raw_auto = scaling_raw.clone();
    scaling_raw_auto.set_requires_grad(true);

    auto activated_auto = torch::exp(scaling_raw_auto);
    auto grad_activated = torch::randn_like(activated_auto);

    activated_auto.backward(grad_activated);
    auto grad_raw_auto = scaling_raw_auto.grad();

    // PATH 2: Manual chain rule
    // Forward: activated = exp(raw)
    // Backward: grad_raw = grad_activated * exp'(raw)
    //         = grad_activated * exp(raw)
    auto activated_manual = torch::exp(scaling_raw);
    auto grad_raw_manual = grad_activated * activated_manual;

    // Compare
    auto diff = (grad_raw_auto - grad_raw_manual).abs().max().item().toFloat();
    auto scale = grad_raw_auto.abs().max().item().toFloat();

    EXPECT_LT(diff / (scale + 1e-8f), tolerance)
        << "Exp gradient mismatch: max_diff=" << diff << ", scale=" << scale;

    std::cout << "Exp: max_diff=" << diff << ", rel_err=" << (diff / (scale + 1e-8f)) << std::endl;
}

// Test normalize activation gradient
TEST_F(ActivationGradientsTest, NormalizeGradient) {
    const float tolerance = 1e-5f;  // Slightly higher tolerance for normalize

    auto rotation_raw = torch::randn({n_points, 4}, torch::kFloat32).cuda();

    // PATH 1: Autograd
    auto rotation_raw_auto = rotation_raw.clone();
    rotation_raw_auto.set_requires_grad(true);

    auto activated_auto = torch::nn::functional::normalize(rotation_raw_auto,
        torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto grad_activated = torch::randn_like(activated_auto);

    activated_auto.backward(grad_activated);
    auto grad_raw_auto = rotation_raw_auto.grad();

    // PATH 2: Manual chain rule
    // Forward: activated = raw / ||raw||
    // Backward: This is complex, involves Jacobian of normalization
    // Formula: grad_raw = (grad_activated - (grad_activated · activated) * activated) / ||raw||
    auto norm = rotation_raw.norm(2, -1, true);  // [n_points, 1]
    auto activated_manual = rotation_raw / norm;

    // Compute dot product: grad_activated · activated
    auto dot_product = (grad_activated * activated_manual).sum(-1, true);  // [n_points, 1]

    // Apply chain rule
    auto grad_raw_manual = (grad_activated - dot_product * activated_manual) / norm;

    // Compare
    auto diff = (grad_raw_auto - grad_raw_manual).abs().max().item().toFloat();
    auto scale = grad_raw_auto.abs().max().item().toFloat();

    EXPECT_LT(diff / (scale + 1e-8f), tolerance)
        << "Normalize gradient mismatch: max_diff=" << diff << ", scale=" << scale;

    std::cout << "Normalize: max_diff=" << diff << ", rel_err=" << (diff / (scale + 1e-8f)) << std::endl;
}

// Test cat activation gradient (trivial - just splits gradient)
TEST_F(ActivationGradientsTest, CatGradient) {
    const float tolerance = 1e-7f;

    int sh0_size = 1;
    int shN_size = 0;  // For degree 0

    auto sh0 = torch::randn({n_points, sh0_size, 3}, torch::kFloat32).cuda();
    auto shN = torch::zeros({n_points, shN_size, 3}, torch::kFloat32).cuda();

    // PATH 1: Autograd
    auto sh0_auto = sh0.clone();
    auto shN_auto = shN.clone();
    sh0_auto.set_requires_grad(true);
    shN_auto.set_requires_grad(true);

    auto activated_auto = torch::cat({sh0_auto, shN_auto}, 1);
    auto grad_activated = torch::randn_like(activated_auto);

    activated_auto.backward(grad_activated);
    auto grad_sh0_auto = sh0_auto.grad();
    auto grad_shN_auto = shN_auto.grad();

    // PATH 2: Manual chain rule
    // Forward: activated = cat(sh0, shN)
    // Backward: Just split the gradient back
    auto grad_sh0_manual = grad_activated.narrow(1, 0, sh0_size);
    torch::Tensor grad_shN_manual;
    if (shN_size > 0) {
        grad_shN_manual = grad_activated.narrow(1, sh0_size, shN_size);
    } else {
        grad_shN_manual = torch::zeros({n_points, 0, 3}, torch::kFloat32).cuda();
    }

    // Compare sh0
    auto diff_sh0 = (grad_sh0_auto - grad_sh0_manual).abs().max().item().toFloat();
    auto scale_sh0 = grad_sh0_auto.abs().max().item().toFloat();

    EXPECT_LT(diff_sh0 / (scale_sh0 + 1e-8f), tolerance)
        << "Cat (sh0) gradient mismatch: max_diff=" << diff_sh0 << ", scale=" << scale_sh0;

    std::cout << "Cat (sh0): max_diff=" << diff_sh0 << ", rel_err=" << (diff_sh0 / (scale_sh0 + 1e-8f)) << std::endl;

    // Compare shN (if it exists)
    if (shN_size > 0 && grad_shN_auto.defined() && grad_shN_manual.defined()) {
        auto diff_shN = (grad_shN_auto - grad_shN_manual).abs().max().item().toFloat();
        auto scale_shN = grad_shN_auto.abs().max().item().toFloat();

        EXPECT_LT(diff_shN / (scale_shN + 1e-8f), tolerance)
            << "Cat (shN) gradient mismatch: max_diff=" << diff_shN << ", scale=" << scale_shN;

        std::cout << "Cat (shN): max_diff=" << diff_shN << ", rel_err=" << (diff_shN / (scale_shN + 1e-8f)) << std::endl;
    }
}

// Test full pipeline with all activations
TEST_F(ActivationGradientsTest, FullPipelineGradient) {
    const float tolerance = 1e-5f;

    // Create all raw parameters
    auto means_raw = torch::randn({n_points, 3}, torch::kFloat32).cuda();
    auto opacity_raw = torch::randn({n_points, 1}, torch::kFloat32).cuda();
    auto scaling_raw = torch::randn({n_points, 3}, torch::kFloat32).cuda();
    auto rotation_raw = torch::randn({n_points, 4}, torch::kFloat32).cuda();
    auto sh0 = torch::randn({n_points, 1, 3}, torch::kFloat32).cuda();
    auto shN = torch::zeros({n_points, 0, 3}, torch::kFloat32).cuda();

    // PATH 1: Autograd
    auto means_auto = means_raw.clone();
    auto opacity_auto = opacity_raw.clone();
    auto scaling_auto = scaling_raw.clone();
    auto rotation_auto = rotation_raw.clone();
    auto sh0_auto = sh0.clone();
    auto shN_auto = shN.clone();

    means_auto.set_requires_grad(true);
    opacity_auto.set_requires_grad(true);
    scaling_auto.set_requires_grad(true);
    rotation_auto.set_requires_grad(true);
    sh0_auto.set_requires_grad(true);
    shN_auto.set_requires_grad(true);

    auto means_activated_auto = means_auto;
    auto opacity_activated_auto = torch::sigmoid(opacity_auto).squeeze(-1);
    auto scaling_activated_auto = torch::exp(scaling_auto);
    auto rotation_activated_auto = torch::nn::functional::normalize(rotation_auto,
        torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto sh_activated_auto = torch::cat({sh0_auto, shN_auto}, 1);

    // Simulate gradients from gsplat backward
    auto grad_means = torch::randn_like(means_activated_auto);
    auto grad_opacity = torch::randn_like(opacity_activated_auto);
    auto grad_scaling = torch::randn_like(scaling_activated_auto);
    auto grad_rotation = torch::randn_like(rotation_activated_auto);
    auto grad_sh = torch::randn_like(sh_activated_auto);

    // Use autograd
    torch::autograd::backward(
        {means_activated_auto, opacity_activated_auto, scaling_activated_auto,
         rotation_activated_auto, sh_activated_auto},
        {grad_means, grad_opacity, grad_scaling, grad_rotation, grad_sh}
    );

    auto grad_means_auto = means_auto.grad();
    auto grad_opacity_auto = opacity_auto.grad();
    auto grad_scaling_auto = scaling_auto.grad();
    auto grad_rotation_auto = rotation_auto.grad();
    auto grad_sh0_auto = sh0_auto.grad();

    // PATH 2: Manual chain rule
    // Means: no activation, gradient passes through
    auto grad_means_manual = grad_means;

    // Opacity: sigmoid
    auto sigmoid_val = torch::sigmoid(opacity_raw);
    auto sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
    auto grad_opacity_manual = grad_opacity.unsqueeze(-1) * sigmoid_deriv;

    // Scaling: exp
    auto scaling_activated = torch::exp(scaling_raw);
    auto grad_scaling_manual = grad_scaling * scaling_activated;

    // Rotation: normalize
    auto norm = rotation_raw.norm(2, -1, true);
    auto rotation_activated = rotation_raw / norm;
    auto dot_product = (grad_rotation * rotation_activated).sum(-1, true);
    auto grad_rotation_manual = (grad_rotation - dot_product * rotation_activated) / norm;

    // SH: cat (just split)
    auto grad_sh0_manual = grad_sh.narrow(1, 0, 1);

    // Compare all gradients
    auto means_diff = (grad_means_auto - grad_means_manual).abs().max().item().toFloat();
    auto opacity_diff = (grad_opacity_auto - grad_opacity_manual).abs().max().item().toFloat();
    auto scaling_diff = (grad_scaling_auto - grad_scaling_manual).abs().max().item().toFloat();
    auto rotation_diff = (grad_rotation_auto - grad_rotation_manual).abs().max().item().toFloat();
    auto sh0_diff = (grad_sh0_auto - grad_sh0_manual).abs().max().item().toFloat();

    auto means_scale = grad_means_auto.abs().max().item().toFloat();
    auto opacity_scale = grad_opacity_auto.abs().max().item().toFloat();
    auto scaling_scale = grad_scaling_auto.abs().max().item().toFloat();
    auto rotation_scale = grad_rotation_auto.abs().max().item().toFloat();
    auto sh0_scale = grad_sh0_auto.abs().max().item().toFloat();

    EXPECT_LT(means_diff / (means_scale + 1e-8f), tolerance);
    EXPECT_LT(opacity_diff / (opacity_scale + 1e-8f), tolerance);
    EXPECT_LT(scaling_diff / (scaling_scale + 1e-8f), tolerance);
    EXPECT_LT(rotation_diff / (rotation_scale + 1e-8f), tolerance);
    EXPECT_LT(sh0_diff / (sh0_scale + 1e-8f), tolerance);

    std::cout << "\nFull pipeline gradient comparison:\n";
    std::cout << "  Means:    rel_err=" << (means_diff / (means_scale + 1e-8f)) << "\n";
    std::cout << "  Opacity:  rel_err=" << (opacity_diff / (opacity_scale + 1e-8f)) << "\n";
    std::cout << "  Scaling:  rel_err=" << (scaling_diff / (scaling_scale + 1e-8f)) << "\n";
    std::cout << "  Rotation: rel_err=" << (rotation_diff / (rotation_scale + 1e-8f)) << "\n";
    std::cout << "  SH0:      rel_err=" << (sh0_diff / (sh0_scale + 1e-8f)) << "\n";
}
