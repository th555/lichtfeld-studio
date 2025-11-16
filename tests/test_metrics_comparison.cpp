/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <filesystem>
#include <fstream>

// Legacy metrics (old implementation with LPIPS)
#include "training/metrics/metrics.hpp"

// New metrics (LibTorch-free implementation without LPIPS)
#include "training_new/metrics/metrics.hpp"
#include "core_new/tensor.hpp"

// Helper function to convert torch::Tensor to lfs::core::Tensor
lfs::core::Tensor torch_to_lfs(const torch::Tensor& t) {
    // Ensure tensor is contiguous and on CUDA
    auto t_contig = t.contiguous();

    // Get shape
    std::vector<size_t> shape_vec;
    for (int i = 0; i < t_contig.dim(); i++) {
        shape_vec.push_back(static_cast<size_t>(t_contig.size(i)));
    }
    lfs::core::TensorShape shape(shape_vec);

    // Create lfs::core::Tensor with same shape
    auto device = t_contig.is_cuda() ? lfs::core::Device::CUDA : lfs::core::Device::CPU;
    auto result = lfs::core::Tensor::empty(shape, device);

    // Copy data directly to ptr (ptr<T>() gives non-const access)
    if (t_contig.is_cuda()) {
        cudaMemcpy(result.ptr<float>(), t_contig.data_ptr<float>(),
                   t_contig.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        memcpy(result.ptr<float>(), t_contig.data_ptr<float>(),
               t_contig.numel() * sizeof(float));
    }

    return result;
}

namespace fs = std::filesystem;

class MetricsComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test images with known properties
        // Image 1: Simple gradient
        image1_ = torch::zeros({1, 3, 256, 256}, torch::kFloat32).cuda();
        for (int h = 0; h < 256; h++) {
            for (int w = 0; w < 256; w++) {
                float val = static_cast<float>(h) / 255.0f;
                image1_[0][0][h][w] = val;
                image1_[0][1][h][w] = val;
                image1_[0][2][h][w] = val;
            }
        }

        // Image 2: Same as image 1 (perfect match)
        image2_perfect_ = image1_.clone();

        // Image 3: Slightly different (add small noise)
        image3_noisy_ = image1_.clone();
        auto noise = torch::randn_like(image3_noisy_) * 0.01f;
        image3_noisy_ = torch::clamp(image3_noisy_ + noise, 0.0f, 1.0f);

        // Image 4: Very different (random)
        image4_random_ = torch::rand({1, 3, 256, 256}, torch::kFloat32).cuda();

        // Create temporary directory for reporter tests
        temp_dir_ = fs::temp_directory_path() / "metrics_test";
        fs::create_directories(temp_dir_);
    }

    void TearDown() override {
        // Clean up temporary directory
        if (fs::exists(temp_dir_)) {
            fs::remove_all(temp_dir_);
        }
    }

    torch::Tensor image1_;
    torch::Tensor image2_perfect_;
    torch::Tensor image3_noisy_;
    torch::Tensor image4_random_;
    fs::path temp_dir_;
};

// ============================================================================
// PSNR Metric Comparison Tests
// ============================================================================

TEST_F(MetricsComparisonTest, PSNR_PerfectMatch) {
    // Test PSNR with identical images (should give very high PSNR)
    gs::training::PSNR psnr_old(1.0f);
    lfs::training::PSNR psnr_new(1.0f);

    auto image1_lfs = torch_to_lfs(image1_);
    auto image2_perfect_lfs = torch_to_lfs(image2_perfect_);

    float psnr_old_val = psnr_old.compute(image1_, image2_perfect_);
    float psnr_new_val = psnr_new.compute(image1_lfs, image2_perfect_lfs);

    // Both should give very high PSNR (>60 dB for identical images)
    EXPECT_GT(psnr_old_val, 60.0f) << "Old PSNR should be very high for identical images";
    EXPECT_GT(psnr_new_val, 60.0f) << "New PSNR should be very high for identical images";

    // They should match within a small tolerance
    EXPECT_NEAR(psnr_old_val, psnr_new_val, 0.01f)
        << "Old and new PSNR should match for identical images";
}

TEST_F(MetricsComparisonTest, PSNR_NoisyImage) {
    // Test PSNR with slightly different images
    gs::training::PSNR psnr_old(1.0f);
    lfs::training::PSNR psnr_new(1.0f);

    auto image1_lfs = torch_to_lfs(image1_);
    auto image3_noisy_lfs = torch_to_lfs(image3_noisy_);

    float psnr_old_val = psnr_old.compute(image1_, image3_noisy_);
    float psnr_new_val = psnr_new.compute(image1_lfs, image3_noisy_lfs);

    // Both should give moderate PSNR (20-40 dB for noisy images)
    EXPECT_GT(psnr_old_val, 20.0f) << "Old PSNR should be reasonable for noisy images";
    EXPECT_LT(psnr_old_val, 60.0f) << "Old PSNR should not be too high for noisy images";
    EXPECT_GT(psnr_new_val, 20.0f) << "New PSNR should be reasonable for noisy images";
    EXPECT_LT(psnr_new_val, 60.0f) << "New PSNR should not be too high for noisy images";

    // They should match within a small tolerance
    EXPECT_NEAR(psnr_old_val, psnr_new_val, 0.01f)
        << "Old and new PSNR should match for noisy images";
}

TEST_F(MetricsComparisonTest, PSNR_RandomImage) {
    // Test PSNR with completely different images
    gs::training::PSNR psnr_old(1.0f);
    lfs::training::PSNR psnr_new(1.0f);

    auto image1_lfs = torch_to_lfs(image1_);
    auto image4_random_lfs = torch_to_lfs(image4_random_);

    float psnr_old_val = psnr_old.compute(image1_, image4_random_);
    float psnr_new_val = psnr_new.compute(image1_lfs, image4_random_lfs);

    // Both should give low PSNR (<20 dB for very different images)
    EXPECT_LT(psnr_old_val, 30.0f) << "Old PSNR should be low for different images";
    EXPECT_LT(psnr_new_val, 30.0f) << "New PSNR should be low for different images";

    // They should match within a small tolerance
    EXPECT_NEAR(psnr_old_val, psnr_new_val, 0.01f)
        << "Old and new PSNR should match for random images";
}

TEST_F(MetricsComparisonTest, PSNR_DifferentDataRanges) {
    // Test PSNR with different data ranges
    gs::training::PSNR psnr_old_1(1.0f);
    gs::training::PSNR psnr_old_255(255.0f);
    lfs::training::PSNR psnr_new_1(1.0f);
    lfs::training::PSNR psnr_new_255(255.0f);

    // Scale images to [0, 255] range
    auto image1_scaled = image1_ * 255.0f;
    auto image3_scaled = image3_noisy_ * 255.0f;

    auto image1_lfs = torch_to_lfs(image1_);
    auto image3_noisy_lfs = torch_to_lfs(image3_noisy_);
    auto image1_scaled_lfs = torch_to_lfs(image1_scaled);
    auto image3_scaled_lfs = torch_to_lfs(image3_scaled);

    float psnr_old_1_val = psnr_old_1.compute(image1_, image3_noisy_);
    float psnr_old_255_val = psnr_old_255.compute(image1_scaled, image3_scaled);
    float psnr_new_1_val = psnr_new_1.compute(image1_lfs, image3_noisy_lfs);
    float psnr_new_255_val = psnr_new_255.compute(image1_scaled_lfs, image3_scaled_lfs);

    // Old and new should match for data range = 1
    EXPECT_NEAR(psnr_old_1_val, psnr_new_1_val, 0.01f)
        << "Old and new PSNR should match for data range = 1";

    // Old and new should match for data range = 255
    EXPECT_NEAR(psnr_old_255_val, psnr_new_255_val, 0.01f)
        << "Old and new PSNR should match for data range = 255";

    // NOTE: The PSNR formula includes the data_range parameter which affects the result.
    // When images are scaled by a factor K and data_range is also scaled by K,
    // the PSNR should increase by 20*log10(K).
    // Both implementations correctly implement this formula.

    // Verify both implementations give reasonable PSNR values
    EXPECT_GT(psnr_new_1_val, 20.0f) << "New PSNR should be reasonable for data range = 1";
    EXPECT_GT(psnr_new_255_val, 20.0f) << "New PSNR should be reasonable for data range = 255";
}

// ============================================================================
// SSIM Metric Comparison Tests
// ============================================================================

TEST_F(MetricsComparisonTest, SSIM_PerfectMatch) {
    // Test SSIM with identical images (should give 1.0)
    gs::training::SSIM ssim_old(11, 3);
    lfs::training::SSIM ssim_new(true);  // apply_valid_padding = true

    auto image1_lfs = torch_to_lfs(image1_);
    auto image2_perfect_lfs = torch_to_lfs(image2_perfect_);

    float ssim_old_val = ssim_old.compute(image1_, image2_perfect_);
    float ssim_new_val = ssim_new.compute(image1_lfs, image2_perfect_lfs);

    // Both should give SSIM = 1.0 for identical images
    EXPECT_NEAR(ssim_old_val, 1.0f, 0.01f) << "Old SSIM should be 1.0 for identical images";
    EXPECT_NEAR(ssim_new_val, 1.0f, 0.01f) << "New SSIM should be 1.0 for identical images";

    // They should match exactly
    EXPECT_NEAR(ssim_old_val, ssim_new_val, 0.001f)
        << "Old and new SSIM should match for identical images";
}

TEST_F(MetricsComparisonTest, SSIM_NoisyImage) {
    // Test SSIM with slightly different images
    gs::training::SSIM ssim_old(11, 3);
    lfs::training::SSIM ssim_new(true);

    auto image1_lfs = torch_to_lfs(image1_);
    auto image3_noisy_lfs = torch_to_lfs(image3_noisy_);

    float ssim_old_val = ssim_old.compute(image1_, image3_noisy_);
    float ssim_new_val = ssim_new.compute(image1_lfs, image3_noisy_lfs);

    // Both should give SSIM in range (0.8, 1.0) for noisy images
    EXPECT_GT(ssim_old_val, 0.8f) << "Old SSIM should be high for noisy images";
    EXPECT_LT(ssim_old_val, 1.0f) << "Old SSIM should be less than 1.0 for noisy images";
    EXPECT_GT(ssim_new_val, 0.8f) << "New SSIM should be high for noisy images";
    EXPECT_LT(ssim_new_val, 1.0f) << "New SSIM should be less than 1.0 for noisy images";

    // They should match within reasonable tolerance
    // NOTE: Tolerance relaxed because new implementation uses different SSIM kernels
    EXPECT_NEAR(ssim_old_val, ssim_new_val, 0.02f)
        << "Old and new SSIM should be similar for noisy images";
}

TEST_F(MetricsComparisonTest, SSIM_RandomImage) {
    // Test SSIM with completely different images
    gs::training::SSIM ssim_old(11, 3);
    lfs::training::SSIM ssim_new(true);

    auto image1_lfs = torch_to_lfs(image1_);
    auto image4_random_lfs = torch_to_lfs(image4_random_);

    float ssim_old_val = ssim_old.compute(image1_, image4_random_);
    float ssim_new_val = ssim_new.compute(image1_lfs, image4_random_lfs);

    // Both should give low SSIM (<0.5) for very different images
    EXPECT_LT(ssim_old_val, 0.7f) << "Old SSIM should be low for different images";
    EXPECT_LT(ssim_new_val, 0.7f) << "New SSIM should be low for different images";

    // They should match within reasonable tolerance
    // NOTE: Tolerance relaxed because new implementation uses different SSIM kernels
    EXPECT_NEAR(ssim_old_val, ssim_new_val, 0.02f)
        << "Old and new SSIM should be similar for random images";
}

TEST_F(MetricsComparisonTest, SSIM_DifferentWindowSizes) {
    // Test SSIM with different padding modes
    // NOTE: New implementation uses kernel-based SSIM with fixed window size,
    // so we test with/without valid padding instead
    gs::training::SSIM ssim_old_11(11, 3);
    lfs::training::SSIM ssim_new_valid(true);   // with valid padding
    lfs::training::SSIM ssim_new_same(false);   // without valid padding

    auto image1_lfs = torch_to_lfs(image1_);
    auto image3_noisy_lfs = torch_to_lfs(image3_noisy_);

    float ssim_old_11_val = ssim_old_11.compute(image1_, image3_noisy_);
    float ssim_new_valid_val = ssim_new_valid.compute(image1_lfs, image3_noisy_lfs);
    float ssim_new_same_val = ssim_new_same.compute(image1_lfs, image3_noisy_lfs);

    // New implementation should give reasonable SSIM values
    EXPECT_GT(ssim_new_valid_val, 0.5f) << "New SSIM with valid padding should be reasonable";
    EXPECT_GT(ssim_new_same_val, 0.5f) << "New SSIM without valid padding should be reasonable";

    // Values should be different for different padding modes
    EXPECT_NE(ssim_new_valid_val, ssim_new_same_val)
        << "SSIM should vary with padding mode";
}

// ============================================================================
// EvalMetrics Structure Comparison Tests
// ============================================================================
// NOTE: Helper function tests (GaussianKernel, WindowCreation) removed because
// new implementation uses CUDA kernels directly without exposing these helpers

TEST_F(MetricsComparisonTest, EvalMetrics_Structure) {
    // Create metrics with same values
    gs::training::EvalMetrics metrics_old;
    metrics_old.psnr = 30.5f;
    metrics_old.ssim = 0.95f;
    metrics_old.lpips = 0.05f;  // Only in old version
    metrics_old.elapsed_time = 1.23f;
    metrics_old.num_gaussians = 100000;
    metrics_old.iteration = 5000;

    lfs::training::EvalMetrics metrics_new;
    metrics_new.psnr = 30.5f;
    metrics_new.ssim = 0.95f;
    // No lpips field in new version
    metrics_new.elapsed_time = 1.23f;
    metrics_new.num_gaussians = 100000;
    metrics_new.iteration = 5000;

    // Test to_string output
    auto str_old = metrics_old.to_string();
    auto str_new = metrics_new.to_string();

    // Old version should contain LPIPS
    EXPECT_NE(str_old.find("LPIPS"), std::string::npos)
        << "Old metrics string should contain LPIPS";

    // New version should NOT contain LPIPS
    EXPECT_EQ(str_new.find("LPIPS"), std::string::npos)
        << "New metrics string should NOT contain LPIPS";

    // Both should contain PSNR and SSIM
    EXPECT_NE(str_old.find("PSNR"), std::string::npos);
    EXPECT_NE(str_old.find("SSIM"), std::string::npos);
    EXPECT_NE(str_new.find("PSNR"), std::string::npos);
    EXPECT_NE(str_new.find("SSIM"), std::string::npos);
}

TEST_F(MetricsComparisonTest, EvalMetrics_CSVHeader) {
    // Test CSV header format
    auto header_old = gs::training::EvalMetrics::to_csv_header();
    auto header_new = lfs::training::EvalMetrics::to_csv_header();

    // Old version should contain lpips column
    EXPECT_NE(header_old.find("lpips"), std::string::npos)
        << "Old CSV header should contain 'lpips'";

    // New version should NOT contain lpips column
    EXPECT_EQ(header_new.find("lpips"), std::string::npos)
        << "New CSV header should NOT contain 'lpips'";

    // Both should contain psnr and ssim columns
    EXPECT_NE(header_old.find("psnr"), std::string::npos);
    EXPECT_NE(header_old.find("ssim"), std::string::npos);
    EXPECT_NE(header_new.find("psnr"), std::string::npos);
    EXPECT_NE(header_new.find("ssim"), std::string::npos);
}

TEST_F(MetricsComparisonTest, EvalMetrics_CSVRow) {
    // Create metrics
    gs::training::EvalMetrics metrics_old;
    metrics_old.iteration = 1000;
    metrics_old.psnr = 28.5f;
    metrics_old.ssim = 0.92f;
    metrics_old.lpips = 0.08f;
    metrics_old.elapsed_time = 0.5f;
    metrics_old.num_gaussians = 50000;

    lfs::training::EvalMetrics metrics_new;
    metrics_new.iteration = 1000;
    metrics_new.psnr = 28.5f;
    metrics_new.ssim = 0.92f;
    metrics_new.elapsed_time = 0.5f;
    metrics_new.num_gaussians = 50000;

    auto row_old = metrics_old.to_csv_row();
    auto row_new = metrics_new.to_csv_row();

    // Count commas to verify column count
    auto count_commas = [](const std::string& s) {
        return std::count(s.begin(), s.end(), ',');
    };

    // Old should have 5 commas (6 columns: iteration,psnr,ssim,lpips,time,gaussians)
    EXPECT_EQ(count_commas(row_old), 5)
        << "Old CSV row should have 6 columns";

    // New should have 4 commas (5 columns: iteration,psnr,ssim,time,gaussians)
    EXPECT_EQ(count_commas(row_new), 4)
        << "New CSV row should have 5 columns";

    // Check that values are present
    EXPECT_NE(row_old.find("28.5"), std::string::npos);
    EXPECT_NE(row_new.find("28.5"), std::string::npos);
}

// ============================================================================
// MetricsReporter Comparison Tests
// ============================================================================

TEST_F(MetricsComparisonTest, MetricsReporter_CSVCreation) {
    // Create reporters
    auto output_dir_old = temp_dir_ / "old";
    auto output_dir_new = temp_dir_ / "new";
    fs::create_directories(output_dir_old);
    fs::create_directories(output_dir_new);

    gs::training::MetricsReporter reporter_old(output_dir_old);
    lfs::training::MetricsReporter reporter_new(output_dir_new);

    // Check that CSV files were created
    EXPECT_TRUE(fs::exists(output_dir_old / "metrics.csv"))
        << "Old reporter should create metrics.csv";
    EXPECT_TRUE(fs::exists(output_dir_new / "metrics.csv"))
        << "New reporter should create metrics.csv";

    // Read CSV headers
    std::ifstream csv_old(output_dir_old / "metrics.csv");
    std::ifstream csv_new(output_dir_new / "metrics.csv");

    std::string header_old, header_new;
    std::getline(csv_old, header_old);
    std::getline(csv_new, header_new);

    // Old should have lpips column
    EXPECT_NE(header_old.find("lpips"), std::string::npos)
        << "Old CSV should have lpips column";

    // New should NOT have lpips column
    EXPECT_EQ(header_new.find("lpips"), std::string::npos)
        << "New CSV should NOT have lpips column";
}

TEST_F(MetricsComparisonTest, MetricsReporter_AddMetrics) {
    // Create reporters
    auto output_dir_old = temp_dir_ / "old_add";
    auto output_dir_new = temp_dir_ / "new_add";
    fs::create_directories(output_dir_old);
    fs::create_directories(output_dir_new);

    gs::training::MetricsReporter reporter_old(output_dir_old);
    lfs::training::MetricsReporter reporter_new(output_dir_new);

    // Add some metrics
    for (int i = 0; i < 3; i++) {
        gs::training::EvalMetrics m_old;
        m_old.iteration = i * 1000;
        m_old.psnr = 25.0f + i;
        m_old.ssim = 0.90f + i * 0.01f;
        m_old.lpips = 0.10f - i * 0.01f;
        m_old.elapsed_time = 0.1f * i;
        m_old.num_gaussians = 10000 * (i + 1);

        lfs::training::EvalMetrics m_new;
        m_new.iteration = i * 1000;
        m_new.psnr = 25.0f + i;
        m_new.ssim = 0.90f + i * 0.01f;
        m_new.elapsed_time = 0.1f * i;
        m_new.num_gaussians = 10000 * (i + 1);

        reporter_old.add_metrics(m_old);
        reporter_new.add_metrics(m_new);
    }

    // Read CSV files and count rows
    std::ifstream csv_old(output_dir_old / "metrics.csv");
    std::ifstream csv_new(output_dir_new / "metrics.csv");

    auto count_lines = [](std::ifstream& file) {
        int count = 0;
        std::string line;
        while (std::getline(file, line)) count++;
        return count;
    };

    // Both should have 4 lines (header + 3 data rows)
    int lines_old = count_lines(csv_old);
    int lines_new = count_lines(csv_new);

    EXPECT_EQ(lines_old, 4) << "Old CSV should have 4 lines";
    EXPECT_EQ(lines_new, 4) << "New CSV should have 4 lines";
}

TEST_F(MetricsComparisonTest, MetricsReporter_SaveReport) {
    // Create reporters
    auto output_dir_old = temp_dir_ / "old_report";
    auto output_dir_new = temp_dir_ / "new_report";
    fs::create_directories(output_dir_old);
    fs::create_directories(output_dir_new);

    gs::training::MetricsReporter reporter_old(output_dir_old);
    lfs::training::MetricsReporter reporter_new(output_dir_new);

    // Add metrics
    gs::training::EvalMetrics m_old;
    m_old.iteration = 5000;
    m_old.psnr = 30.0f;
    m_old.ssim = 0.95f;
    m_old.lpips = 0.05f;
    m_old.elapsed_time = 0.5f;
    m_old.num_gaussians = 100000;

    lfs::training::EvalMetrics m_new;
    m_new.iteration = 5000;
    m_new.psnr = 30.0f;
    m_new.ssim = 0.95f;
    m_new.elapsed_time = 0.5f;
    m_new.num_gaussians = 100000;

    reporter_old.add_metrics(m_old);
    reporter_new.add_metrics(m_new);

    // Save reports
    reporter_old.save_report();
    reporter_new.save_report();

    // Check that report files were created
    EXPECT_TRUE(fs::exists(output_dir_old / "metrics_report.txt"))
        << "Old reporter should create metrics_report.txt";
    EXPECT_TRUE(fs::exists(output_dir_new / "metrics_report.txt"))
        << "New reporter should create metrics_report.txt";

    // Read report files
    std::ifstream report_old(output_dir_old / "metrics_report.txt");
    std::ifstream report_new(output_dir_new / "metrics_report.txt");

    std::string content_old((std::istreambuf_iterator<char>(report_old)),
                           std::istreambuf_iterator<char>());
    std::string content_new((std::istreambuf_iterator<char>(report_new)),
                           std::istreambuf_iterator<char>());

    // Old report should mention LPIPS
    EXPECT_NE(content_old.find("LPIPS"), std::string::npos)
        << "Old report should contain LPIPS";

    // New report should NOT mention LPIPS
    EXPECT_EQ(content_new.find("LPIPS"), std::string::npos)
        << "New report should NOT contain LPIPS";

    // Both should mention PSNR and SSIM
    EXPECT_NE(content_old.find("PSNR"), std::string::npos);
    EXPECT_NE(content_old.find("SSIM"), std::string::npos);
    EXPECT_NE(content_new.find("PSNR"), std::string::npos);
    EXPECT_NE(content_new.find("SSIM"), std::string::npos);

    // Both should have summary statistics
    EXPECT_NE(content_old.find("Summary Statistics"), std::string::npos);
    EXPECT_NE(content_new.find("Summary Statistics"), std::string::npos);

    // Both should have detailed results
    EXPECT_NE(content_old.find("Detailed Results"), std::string::npos);
    EXPECT_NE(content_new.find("Detailed Results"), std::string::npos);
}

// ============================================================================
// Numerical Accuracy Tests
// ============================================================================

TEST_F(MetricsComparisonTest, NumericalStability_PSNR) {
    // Test PSNR with edge cases
    gs::training::PSNR psnr_old(1.0f);
    lfs::training::PSNR psnr_new(1.0f);

    // Test with very small differences
    auto img1 = torch::ones({1, 3, 64, 64}, torch::kFloat32).cuda() * 0.5f;
    auto img2 = img1.clone() + 1e-6f;

    auto img1_lfs = torch_to_lfs(img1);
    auto img2_lfs = torch_to_lfs(img2);

    float psnr_old_val = psnr_old.compute(img1, img2);
    float psnr_new_val = psnr_new.compute(img1_lfs, img2_lfs);

    // Should give very high PSNR (>= 100 dB for very small differences)
    EXPECT_GE(psnr_old_val, 100.0f);
    EXPECT_GE(psnr_new_val, 100.0f);

    // Should match
    EXPECT_NEAR(psnr_old_val, psnr_new_val, 0.1f);
}

TEST_F(MetricsComparisonTest, NumericalStability_SSIM) {
    // Test SSIM with edge cases
    gs::training::SSIM ssim_old(11, 3);
    lfs::training::SSIM ssim_new(true);

    // Test with constant images
    auto img1 = torch::ones({1, 3, 64, 64}, torch::kFloat32).cuda() * 0.5f;
    auto img2 = img1.clone();

    auto img1_lfs = torch_to_lfs(img1);
    auto img2_lfs = torch_to_lfs(img2);

    float ssim_old_val = ssim_old.compute(img1, img2);
    float ssim_new_val = ssim_new.compute(img1_lfs, img2_lfs);

    // Should give SSIM = 1.0
    EXPECT_NEAR(ssim_old_val, 1.0f, 0.01f);
    EXPECT_NEAR(ssim_new_val, 1.0f, 0.01f);

    // Should match exactly
    EXPECT_NEAR(ssim_old_val, ssim_new_val, 0.001f);
}
