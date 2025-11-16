// Test to compare PLY loading between legacy (LibTorch) and new (LibTorch-free) loaders
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <cmath>

// Legacy loader
#include "loader/formats/ply.hpp"

// New loader
#include "loader_new/formats/ply.hpp"

TEST(PLYLoadingTest, CompareLoadersOnSplat30k) {
    const char* ply_path = "/home/paja/projects/gaussian-splatting-cuda/output/splat_30000.ply";

    spdlog::info("=== Loading PLY with LEGACY loader ===");
    auto legacy_result = gs::loader::load_ply(ply_path);
    ASSERT_TRUE(legacy_result.has_value()) << "Legacy loader failed: " << legacy_result.error();
    auto& legacy_splat = legacy_result.value();

    spdlog::info("Legacy: Loaded {} Gaussians", legacy_splat.size());
    spdlog::info("Legacy: Scene scale = {}", legacy_splat.get_scene_scale());

    // Get legacy tensors (move to CPU for comparison)
    auto legacy_means = legacy_splat.means().cpu();
    auto legacy_opacity = legacy_splat.opacity_raw().cpu();
    auto legacy_scaling = legacy_splat.scaling_raw().cpu();
    auto legacy_rotation = legacy_splat.rotation_raw().cpu();
    auto legacy_sh0 = legacy_splat.sh0().cpu();
    auto legacy_shN = legacy_splat.shN().cpu();

    spdlog::info("Legacy shapes:");
    spdlog::info("  means: [{}, {}]", legacy_means.size(0), legacy_means.size(1));
    spdlog::info("  opacity_raw: [{}{}]", legacy_opacity.size(0),
                 legacy_opacity.dim() > 1 ? fmt::format(", {}", legacy_opacity.size(1)) : "");
    spdlog::info("  scaling_raw: [{}, {}]", legacy_scaling.size(0), legacy_scaling.size(1));
    spdlog::info("  rotation_raw: [{}, {}]", legacy_rotation.size(0), legacy_rotation.size(1));
    spdlog::info("  sh0: [{}, {}, {}]", legacy_sh0.size(0), legacy_sh0.size(1), legacy_sh0.size(2));
    spdlog::info("  shN: [{}, {}, {}]", legacy_shN.size(0), legacy_shN.size(1), legacy_shN.size(2));

    spdlog::info("=== Loading PLY with NEW loader ===");
    auto new_result = lfs::loader::load_ply(ply_path);
    ASSERT_TRUE(new_result.has_value()) << "New loader failed: " << new_result.error();
    auto& new_splat = new_result.value();

    spdlog::info("New: Loaded {} Gaussians", new_splat.size());
    spdlog::info("New: Scene scale = {}", new_splat.get_scene_scale());

    // Get new tensors (move to CPU for comparison)
    auto new_means_cpu = new_splat.means().to(lfs::core::Device::CPU);
    auto new_opacity_cpu = new_splat.opacity_raw().to(lfs::core::Device::CPU);
    auto new_scaling_cpu = new_splat.scaling_raw().to(lfs::core::Device::CPU);
    auto new_rotation_cpu = new_splat.rotation_raw().to(lfs::core::Device::CPU);
    auto new_sh0_cpu = new_splat.sh0().to(lfs::core::Device::CPU);
    auto new_shN_cpu = new_splat.shN().to(lfs::core::Device::CPU);

    spdlog::info("New shapes:");
    spdlog::info("  means: [{}, {}]", new_means_cpu.shape()[0], new_means_cpu.shape()[1]);

    if (new_opacity_cpu.ndim() == 1) {
        spdlog::info("  opacity_raw: [{}]", new_opacity_cpu.shape()[0]);
    } else {
        spdlog::info("  opacity_raw: [{}, {}]", new_opacity_cpu.shape()[0], new_opacity_cpu.shape()[1]);
    }

    spdlog::info("  scaling_raw: [{}, {}]", new_scaling_cpu.shape()[0], new_scaling_cpu.shape()[1]);
    spdlog::info("  rotation_raw: [{}, {}]", new_rotation_cpu.shape()[0], new_rotation_cpu.shape()[1]);
    spdlog::info("  sh0: [{}, {}, {}]", new_sh0_cpu.shape()[0], new_sh0_cpu.shape()[1], new_sh0_cpu.shape()[2]);
    spdlog::info("  shN: [{}, {}, {}]", new_shN_cpu.shape()[0], new_shN_cpu.shape()[1], new_shN_cpu.shape()[2]);

    // Compare counts
    EXPECT_EQ(legacy_splat.size(), new_splat.size()) << "Gaussian count mismatch!";

    // Compare scene scales
    EXPECT_NEAR(legacy_splat.get_scene_scale(), new_splat.get_scene_scale(), 1e-4)
        << "Scene scale mismatch!";

    // Get raw pointers
    const float* legacy_means_ptr = legacy_means.template data_ptr<float>();
    const float* legacy_opacity_ptr = legacy_opacity.template data_ptr<float>();
    const float* legacy_scaling_ptr = legacy_scaling.template data_ptr<float>();
    const float* legacy_rotation_ptr = legacy_rotation.template data_ptr<float>();
    const float* legacy_sh0_ptr = legacy_sh0.template data_ptr<float>();
    const float* legacy_shN_ptr = legacy_shN.template data_ptr<float>();

    const float* new_means_ptr = new_means_cpu.template ptr<float>();
    const float* new_opacity_ptr = new_opacity_cpu.template ptr<float>();
    const float* new_scaling_ptr = new_scaling_cpu.template ptr<float>();
    const float* new_rotation_ptr = new_rotation_cpu.template ptr<float>();
    const float* new_sh0_ptr = new_sh0_cpu.template ptr<float>();
    const float* new_shN_ptr = new_shN_cpu.template ptr<float>();

    size_t n = legacy_splat.size();

    spdlog::info("=== Comparing first 10 Gaussians ===");
    for (int i = 0; i < std::min(10, static_cast<int>(n)); ++i) {
        spdlog::info("Gaussian {}:", i);

        // Means
        float means_diff = 0.0f;
        for (int j = 0; j < 3; ++j) {
            float diff = std::abs(legacy_means_ptr[i*3+j] - new_means_ptr[i*3+j]);
            means_diff = std::max(means_diff, diff);
        }
        spdlog::info("  means: Legacy=[{:.6f}, {:.6f}, {:.6f}] New=[{:.6f}, {:.6f}, {:.6f}] MaxDiff={:.6e}",
                     legacy_means_ptr[i*3], legacy_means_ptr[i*3+1], legacy_means_ptr[i*3+2],
                     new_means_ptr[i*3], new_means_ptr[i*3+1], new_means_ptr[i*3+2], means_diff);

        // Opacity (handle shape difference)
        float legacy_op = legacy_opacity_ptr[i];
        float new_op = new_opacity_ptr[i];
        float op_diff = std::abs(legacy_op - new_op);
        spdlog::info("  opacity_raw: Legacy={:.6f} New={:.6f} Diff={:.6e}",
                     legacy_op, new_op, op_diff);

        // Scaling
        float scaling_diff = 0.0f;
        for (int j = 0; j < 3; ++j) {
            float diff = std::abs(legacy_scaling_ptr[i*3+j] - new_scaling_ptr[i*3+j]);
            scaling_diff = std::max(scaling_diff, diff);
        }
        spdlog::info("  scaling_raw: Legacy=[{:.6f}, {:.6f}, {:.6f}] New=[{:.6f}, {:.6f}, {:.6f}] MaxDiff={:.6e}",
                     legacy_scaling_ptr[i*3], legacy_scaling_ptr[i*3+1], legacy_scaling_ptr[i*3+2],
                     new_scaling_ptr[i*3], new_scaling_ptr[i*3+1], new_scaling_ptr[i*3+2], scaling_diff);

        // Rotation
        float rotation_diff = 0.0f;
        for (int j = 0; j < 4; ++j) {
            float diff = std::abs(legacy_rotation_ptr[i*4+j] - new_rotation_ptr[i*4+j]);
            rotation_diff = std::max(rotation_diff, diff);
        }
        spdlog::info("  rotation_raw: Legacy=[{:.6f}, {:.6f}, {:.6f}, {:.6f}] New=[{:.6f}, {:.6f}, {:.6f}, {:.6f}] MaxDiff={:.6e}",
                     legacy_rotation_ptr[i*4], legacy_rotation_ptr[i*4+1], legacy_rotation_ptr[i*4+2], legacy_rotation_ptr[i*4+3],
                     new_rotation_ptr[i*4], new_rotation_ptr[i*4+1], new_rotation_ptr[i*4+2], new_rotation_ptr[i*4+3], rotation_diff);
    }

    // Statistical comparison across all Gaussians
    spdlog::info("=== Statistical comparison across all {} Gaussians ===", n);

    // Means
    double means_max_diff = 0.0, means_sum_diff = 0.0;
    for (size_t i = 0; i < n * 3; ++i) {
        double diff = std::abs(legacy_means_ptr[i] - new_means_ptr[i]);
        means_max_diff = std::max(means_max_diff, diff);
        means_sum_diff += diff;
    }
    spdlog::info("Means: Max diff = {:.6e}, Mean diff = {:.6e}",
                 means_max_diff, means_sum_diff / (n * 3));

    // Opacity
    double opacity_max_diff = 0.0, opacity_sum_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(legacy_opacity_ptr[i] - new_opacity_ptr[i]);
        opacity_max_diff = std::max(opacity_max_diff, diff);
        opacity_sum_diff += diff;
    }
    spdlog::info("Opacity: Max diff = {:.6e}, Mean diff = {:.6e}",
                 opacity_max_diff, opacity_sum_diff / n);

    // Scaling
    double scaling_max_diff = 0.0, scaling_sum_diff = 0.0;
    for (size_t i = 0; i < n * 3; ++i) {
        double diff = std::abs(legacy_scaling_ptr[i] - new_scaling_ptr[i]);
        scaling_max_diff = std::max(scaling_max_diff, diff);
        scaling_sum_diff += diff;
    }
    spdlog::info("Scaling: Max diff = {:.6e}, Mean diff = {:.6e}",
                 scaling_max_diff, scaling_sum_diff / (n * 3));

    // Rotation
    double rotation_max_diff = 0.0, rotation_sum_diff = 0.0;
    for (size_t i = 0; i < n * 4; ++i) {
        double diff = std::abs(legacy_rotation_ptr[i] - new_rotation_ptr[i]);
        rotation_max_diff = std::max(rotation_max_diff, diff);
        rotation_sum_diff += diff;
    }
    spdlog::info("Rotation: Max diff = {:.6e}, Mean diff = {:.6e}",
                 rotation_max_diff, rotation_sum_diff / (n * 4));

    // Assertions - values should be identical or very close
    EXPECT_LT(means_max_diff, 1e-4) << "Means differ significantly!";
    EXPECT_LT(opacity_max_diff, 1e-4) << "Opacity differs significantly!";
    EXPECT_LT(scaling_max_diff, 1e-4) << "Scaling differs significantly!";
    EXPECT_LT(rotation_max_diff, 1e-4) << "Rotation differs significantly!";
}
