// Test to compare PLY saving between legacy (LibTorch) and new (LibTorch-free) implementations
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <cmath>

// Legacy
#include "core/splat_data.hpp"
#include "loader/formats/ply.hpp"

// New
#include "core_new/splat_data.hpp"
#include "loader_new/formats/ply.hpp"

TEST(PLYSaveTest, CompareSaveImplementations) {
    const char* input_ply = "/home/paja/projects/gaussian-splatting-cuda/output/splat_30000.ply";

    spdlog::info("=== Loading with LEGACY ===");
    auto legacy_load_result = gs::loader::load_ply(input_ply);
    ASSERT_TRUE(legacy_load_result.has_value()) << "Legacy load failed: " << legacy_load_result.error();
    auto& legacy_splat = legacy_load_result.value();

    spdlog::info("Legacy loaded: {} Gaussians", legacy_splat.size());

    spdlog::info("=== Loading with NEW ===");
    auto new_load_result = lfs::loader::load_ply(input_ply);
    ASSERT_TRUE(new_load_result.has_value()) << "New load failed: " << new_load_result.error();
    auto& new_splat = new_load_result.value();

    spdlog::info("New loaded: {} Gaussians", new_splat.size());

    spdlog::info("=== Comparing PointCloud objects (before save) ===");

    // Convert to PointCloud (what gets saved)
    auto legacy_pc = legacy_splat.to_point_cloud();
    auto new_pc = new_splat.to_point_cloud();

    // Get PointCloud tensors
    auto legacy_means = legacy_pc.means.cpu();
    auto legacy_opacity = legacy_pc.opacity.cpu();
    auto legacy_scaling = legacy_pc.scaling.cpu();
    auto legacy_rotation = legacy_pc.rotation.cpu();
    auto legacy_sh0 = legacy_pc.sh0.cpu();
    auto legacy_shN = legacy_pc.shN.cpu();

    auto new_means = new_pc.means.to(lfs::core::Device::CPU);
    auto new_opacity = new_pc.opacity.to(lfs::core::Device::CPU);
    auto new_scaling = new_pc.scaling.to(lfs::core::Device::CPU);
    auto new_rotation = new_pc.rotation.to(lfs::core::Device::CPU);
    auto new_sh0 = new_pc.sh0.to(lfs::core::Device::CPU);
    auto new_shN = new_pc.shN.to(lfs::core::Device::CPU);

    spdlog::info("Legacy PointCloud shapes:");
    spdlog::info("  means: [{}, {}]", legacy_means.size(0), legacy_means.size(1));
    spdlog::info("  opacity: [{}{}]", legacy_opacity.size(0),
                 legacy_opacity.dim() > 1 ? fmt::format(", {}", legacy_opacity.size(1)) : "");
    spdlog::info("  scaling: [{}, {}]", legacy_scaling.size(0), legacy_scaling.size(1));
    spdlog::info("  rotation: [{}, {}]", legacy_rotation.size(0), legacy_rotation.size(1));
    spdlog::info("  sh0: [{}{}]", legacy_sh0.size(0),
                 legacy_sh0.dim() > 1 ? fmt::format(", {}", legacy_sh0.size(1)) : "");
    spdlog::info("  shN: [{}{}]", legacy_shN.size(0),
                 legacy_shN.dim() > 1 ? fmt::format(", {}", legacy_shN.size(1)) : "");

    spdlog::info("New PointCloud shapes:");
    spdlog::info("  means: [{}, {}]", new_means.shape()[0], new_means.shape()[1]);
    spdlog::info("  opacity: [{}{}]", new_opacity.shape()[0],
                 new_opacity.ndim() > 1 ? fmt::format(", {}", new_opacity.shape()[1]) : "");
    spdlog::info("  scaling: [{}, {}]", new_scaling.shape()[0], new_scaling.shape()[1]);
    spdlog::info("  rotation: [{}, {}]", new_rotation.shape()[0], new_rotation.shape()[1]);
    spdlog::info("  sh0: [{}{}]", new_sh0.shape()[0],
                 new_sh0.ndim() > 1 ? fmt::format(", {}", new_sh0.shape()[1]) : "");
    spdlog::info("  shN: [{}{}]", new_shN.shape()[0],
                 new_shN.ndim() > 1 ? fmt::format(", {}", new_shN.shape()[1]) : "");

    // Compare first 5 Gaussians element by element
    spdlog::info("=== Comparing first 5 Gaussians ===");

    const float* legacy_means_ptr = legacy_means.template data_ptr<float>();
    const float* legacy_opacity_ptr = legacy_opacity.template data_ptr<float>();
    const float* legacy_scaling_ptr = legacy_scaling.template data_ptr<float>();
    const float* legacy_rotation_ptr = legacy_rotation.template data_ptr<float>();
    const float* legacy_sh0_ptr = legacy_sh0.template data_ptr<float>();
    const float* legacy_shN_ptr = legacy_shN.template data_ptr<float>();

    const float* new_means_ptr = new_means.template ptr<float>();
    const float* new_opacity_ptr = new_opacity.template ptr<float>();
    const float* new_scaling_ptr = new_scaling.template ptr<float>();
    const float* new_rotation_ptr = new_rotation.template ptr<float>();
    const float* new_sh0_ptr = new_sh0.template ptr<float>();
    const float* new_shN_ptr = new_shN.template ptr<float>();

    size_t n = legacy_splat.size();

    for (int i = 0; i < std::min(5, static_cast<int>(n)); ++i) {
        spdlog::info("Gaussian {}:", i);

        // Means
        spdlog::info("  means: Legacy=[{:.6f}, {:.6f}, {:.6f}] New=[{:.6f}, {:.6f}, {:.6f}]",
                     legacy_means_ptr[i*3], legacy_means_ptr[i*3+1], legacy_means_ptr[i*3+2],
                     new_means_ptr[i*3], new_means_ptr[i*3+1], new_means_ptr[i*3+2]);

        // Opacity
        float legacy_op = legacy_opacity.dim() > 1 ? legacy_opacity_ptr[i] : legacy_opacity_ptr[i];
        float new_op = new_opacity.ndim() > 1 ? new_opacity_ptr[i] : new_opacity_ptr[i];
        spdlog::info("  opacity: Legacy={:.6f} New={:.6f}", legacy_op, new_op);

        // Scaling
        spdlog::info("  scaling: Legacy=[{:.6f}, {:.6f}, {:.6f}] New=[{:.6f}, {:.6f}, {:.6f}]",
                     legacy_scaling_ptr[i*3], legacy_scaling_ptr[i*3+1], legacy_scaling_ptr[i*3+2],
                     new_scaling_ptr[i*3], new_scaling_ptr[i*3+1], new_scaling_ptr[i*3+2]);

        // Rotation
        spdlog::info("  rotation: Legacy=[{:.6f}, {:.6f}, {:.6f}, {:.6f}] New=[{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                     legacy_rotation_ptr[i*4], legacy_rotation_ptr[i*4+1],
                     legacy_rotation_ptr[i*4+2], legacy_rotation_ptr[i*4+3],
                     new_rotation_ptr[i*4], new_rotation_ptr[i*4+1],
                     new_rotation_ptr[i*4+2], new_rotation_ptr[i*4+3]);

        // SH0 (first 3 values)
        int sh0_stride = legacy_sh0.dim() > 1 ? legacy_sh0.size(1) : 1;
        int new_sh0_stride = new_sh0.ndim() > 1 ? new_sh0.shape()[1] : 1;
        spdlog::info("  sh0[0:3]: Legacy=[{:.6f}, {:.6f}, {:.6f}] New=[{:.6f}, {:.6f}, {:.6f}]",
                     legacy_sh0_ptr[i*sh0_stride], legacy_sh0_ptr[i*sh0_stride+1], legacy_sh0_ptr[i*sh0_stride+2],
                     new_sh0_ptr[i*new_sh0_stride], new_sh0_ptr[i*new_sh0_stride+1], new_sh0_ptr[i*new_sh0_stride+2]);
    }

    // Just check shapes match for now
    EXPECT_EQ(legacy_means.size(0), new_means.shape()[0]);
    EXPECT_EQ(legacy_means.size(1), new_means.shape()[1]);
}
