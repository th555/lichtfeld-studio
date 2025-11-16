/* Benchmark: Slice Assignment Performance vs LibTorch
 *
 * Fair comparison of slice assignment operations between our custom Tensor library
 * and LibTorch, using identical data and operations.
 */

#include "core_new/logger.hpp"
#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <torch/torch.h>

using lfs::core::Device;
using lfs::core::Tensor;

class SliceAssignmentBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        // Warm up GPU
        auto warmup = Tensor::ones({1000, 1000}, Device::CUDA);
        warmup.cpu();
        torch::cuda::synchronize();
    }

    // Helper to measure time in microseconds
    template <typename Func>
    double measure_time_us(Func&& func, int iterations = 100) {
        // Warm up
        func();
        torch::cuda::synchronize();

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        torch::cuda::synchronize();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) / iterations;
    }

    void print_comparison(const std::string& test_name, double lfs_time_us, double torch_time_us,
                          size_t data_size_mb) {
        double speedup = torch_time_us / lfs_time_us;
        double lfs_bandwidth = (data_size_mb / (lfs_time_us / 1e6));
        double torch_bandwidth = (data_size_mb / (torch_time_us / 1e6));

        std::cout << "\n"
                  << test_name << "\n";
        std::cout << "  LFS Tensor:  " << std::fixed << std::setprecision(2)
                  << lfs_time_us << " μs  (" << lfs_bandwidth << " MB/s)\n";
        std::cout << "  LibTorch:    " << torch_time_us << " μs  ("
                  << torch_bandwidth << " MB/s)\n";
        std::cout << "  Speedup:     " << speedup << "x ";
        if (speedup > 1.0) {
            std::cout << "(LFS faster)";
        } else if (speedup < 1.0) {
            std::cout << "(LibTorch faster)";
        } else {
            std::cout << "(same)";
        }
        std::cout << "\n";
    }
};

// =============================================================================
// Small Tensor Benchmarks (Latency focused)
// =============================================================================

TEST_F(SliceAssignmentBenchmark, SmallTensor_SimpleSlice) {
    const size_t N = 1000;
    const size_t slice_size = 100;
    const size_t data_mb = (slice_size * 3 * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_size, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        lfs_dst.slice(0, 0, slice_size) = lfs_src;
    });

    // LibTorch
    auto torch_dst = torch::zeros({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_size, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        torch_dst.slice(0, 0, slice_size) = torch_src;
    });

    print_comparison("Small Tensor (1K elements, 100 slice)", lfs_time, torch_time, data_mb);
}

TEST_F(SliceAssignmentBenchmark, SmallTensor_OffsetSlice) {
    const size_t N = 1000;
    const size_t slice_size = 100;
    const size_t offset = 500;
    const size_t data_mb = (slice_size * 3 * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_size, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        lfs_dst.slice(0, offset, offset + slice_size) = lfs_src;
    });

    // LibTorch
    auto torch_dst = torch::zeros({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_size, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        torch_dst.slice(0, offset, offset + slice_size) = torch_src;
    });

    print_comparison("Small Tensor Offset Slice (offset=500)", lfs_time, torch_time, data_mb);
}

TEST_F(SliceAssignmentBenchmark, SmallTensor_MultipleSequential) {
    const size_t N = 1000;
    const size_t slice_size = 50;
    const size_t data_mb = (slice_size * 3 * 10 * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_size, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        for (size_t i = 0; i < 10; ++i) {
            lfs_dst.slice(0, i * slice_size, (i + 1) * slice_size) = lfs_src;
        }
    },
                                    10); // Fewer iterations since each run does 10 operations

    // LibTorch
    auto torch_dst = torch::zeros({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_size, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        for (size_t i = 0; i < 10; ++i) {
            torch_dst.slice(0, i * slice_size, (i + 1) * slice_size) = torch_src;
        }
    },
                                      10);

    print_comparison("Small Tensor Multiple Sequential (10x)", lfs_time, torch_time, data_mb);
}

// =============================================================================
// Medium Tensor Benchmarks (Typical workload)
// =============================================================================

TEST_F(SliceAssignmentBenchmark, MediumTensor_100K) {
    const size_t N = 100000;
    const size_t slice_size = 10000;
    const size_t data_mb = (slice_size * 3 * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_size, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        lfs_dst.slice(0, 0, slice_size) = lfs_src;
    });

    // LibTorch
    auto torch_dst = torch::zeros({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_size, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        torch_dst.slice(0, 0, slice_size) = torch_src;
    });

    print_comparison("Medium Tensor (100K elements, 10K slice)", lfs_time, torch_time, data_mb);
}

TEST_F(SliceAssignmentBenchmark, MediumTensor_ViewToView) {
    const size_t N = 100000;
    const size_t slice_size = 10000;
    const size_t data_mb = (slice_size * 3 * sizeof(float)) / (1024 * 1024);

    // LFS Tensor - view to view assignment
    Tensor lfs_dst = Tensor::zeros({N, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({N * 2, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        lfs_dst.slice(0, 0, slice_size) = lfs_src.slice(0, 5000, 5000 + slice_size);
    });

    // LibTorch
    auto torch_dst = torch::zeros({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({N * 2, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        torch_dst.slice(0, 0, slice_size) = torch_src.slice(0, 5000, 5000 + slice_size);
    });

    print_comparison("Medium Tensor View-to-View", lfs_time, torch_time, data_mb);
}

// =============================================================================
// Large Tensor Benchmarks (Bandwidth focused)
// =============================================================================

TEST_F(SliceAssignmentBenchmark, LargeTensor_1M) {
    const size_t N = 1000000;
    const size_t slice_size = 100000;
    const size_t data_mb = (slice_size * 3 * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_size, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        lfs_dst.slice(0, 0, slice_size) = lfs_src;
    },
                                    50); // Fewer iterations for large tensors

    // LibTorch
    auto torch_dst = torch::zeros({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_size, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        torch_dst.slice(0, 0, slice_size) = torch_src;
    },
                                      50);

    print_comparison("Large Tensor (1M elements, 100K slice)", lfs_time, torch_time, data_mb);
}

TEST_F(SliceAssignmentBenchmark, LargeTensor_10M) {
    const size_t N = 10000000;
    const size_t slice_size = 1000000;
    const size_t data_mb = (slice_size * 3 * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_size, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        lfs_dst.slice(0, 0, slice_size) = lfs_src;
    },
                                    20);

    // LibTorch
    auto torch_dst = torch::zeros({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_size, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        torch_dst.slice(0, 0, slice_size) = torch_src;
    },
                                      20);

    print_comparison("Large Tensor (10M elements, 1M slice)", lfs_time, torch_time, data_mb);
}

TEST_F(SliceAssignmentBenchmark, LargeTensor_HalfAssignment) {
    const size_t N = 10000000;
    const size_t slice_size = N / 2;
    const size_t data_mb = (slice_size * 3 * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_size, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        lfs_dst.slice(0, 0, slice_size) = lfs_src;
    },
                                    20);

    // LibTorch
    auto torch_dst = torch::zeros({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_size, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        torch_dst.slice(0, 0, slice_size) = torch_src;
    },
                                      20);

    print_comparison("Large Tensor Half Assignment (10M -> 5M)", lfs_time, torch_time, data_mb);
}

// =============================================================================
// 3D Tensor Benchmarks (Realistic scene operations)
// =============================================================================

TEST_F(SliceAssignmentBenchmark, Tensor3D_GaussianSplatScenario) {
    // Realistic Gaussian Splatting scenario: [N, 1, 3] for sh0 coefficients
    const size_t N = 1000000;
    const size_t slice_size = 100000;
    const size_t data_mb = (slice_size * 1 * 3 * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 1, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_size, 1, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        lfs_dst.slice(0, 0, slice_size) = lfs_src;
    },
                                    50);

    // LibTorch
    auto torch_dst = torch::zeros({N, 1, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_size, 1, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        torch_dst.slice(0, 0, slice_size) = torch_src;
    },
                                      50);

    print_comparison("3D Tensor Gaussian Splat sh0 (1M x 1 x 3)", lfs_time, torch_time, data_mb);
}

TEST_F(SliceAssignmentBenchmark, Tensor3D_HighDimSH) {
    // Higher order SH coefficients: [N, 15, 3] for degree 3
    const size_t N = 1000000;
    const size_t slice_size = 100000;
    const size_t data_mb = (slice_size * 15 * 3 * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 15, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_size, 15, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        lfs_dst.slice(0, 0, slice_size) = lfs_src;
    },
                                    30);

    // LibTorch
    auto torch_dst = torch::zeros({N, 15, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_size, 15, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        torch_dst.slice(0, 0, slice_size) = torch_src;
    },
                                      30);

    print_comparison("3D Tensor High-Dim SH (1M x 15 x 3)", lfs_time, torch_time, data_mb);
}

TEST_F(SliceAssignmentBenchmark, Tensor3D_MultiDimSlice) {
    // Slicing in multiple dimensions
    const size_t N = 100000;
    const size_t slice_dim0 = 10000;
    const size_t slice_dim1 = 8;
    const size_t data_mb = (slice_dim0 * slice_dim1 * 3 * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 15, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_dim0, slice_dim1, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        lfs_dst.slice(0, 0, slice_dim0).slice(1, 0, slice_dim1) = lfs_src;
    });

    // LibTorch
    auto torch_dst = torch::zeros({N, 15, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_dim0, slice_dim1, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        torch_dst.slice(0, 0, slice_dim0).slice(1, 0, slice_dim1) = torch_src;
    });

    print_comparison("3D Tensor Multi-Dim Slice", lfs_time, torch_time, data_mb);
}

// =============================================================================
// Worst Case Scenarios (Stress testing)
// =============================================================================

TEST_F(SliceAssignmentBenchmark, WorstCase_TinySlices) {
    // Many tiny slice operations
    const size_t N = 10000;
    const size_t slice_size = 10;
    const size_t num_ops = 100;
    const size_t data_mb = (slice_size * 3 * num_ops * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_size, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        for (size_t i = 0; i < num_ops; ++i) {
            lfs_dst.slice(0, i * slice_size, (i + 1) * slice_size) = lfs_src;
        }
    },
                                    5);

    // LibTorch
    auto torch_dst = torch::zeros({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_size, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        for (size_t i = 0; i < num_ops; ++i) {
            torch_dst.slice(0, i * slice_size, (i + 1) * slice_size) = torch_src;
        }
    },
                                      5);

    print_comparison("Worst Case: Tiny Slices (100 x 10 elements)", lfs_time, torch_time, data_mb);
}

TEST_F(SliceAssignmentBenchmark, WorstCase_HighlyFragmented) {
    // Fragmented access pattern
    const size_t N = 1000000;
    const size_t slice_size = 1000;
    const size_t stride = 10000;
    const size_t num_ops = 10;
    const size_t data_mb = (slice_size * 3 * num_ops * sizeof(float)) / (1024 * 1024);

    // LFS Tensor
    Tensor lfs_dst = Tensor::zeros({N, 3}, Device::CUDA);
    Tensor lfs_src = Tensor::ones({slice_size, 3}, Device::CUDA);

    auto lfs_time = measure_time_us([&]() {
        for (size_t i = 0; i < num_ops; ++i) {
            size_t offset = i * stride;
            lfs_dst.slice(0, offset, offset + slice_size) = lfs_src;
        }
    },
                                    20);

    // LibTorch
    auto torch_dst = torch::zeros({N, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_src = torch::ones({slice_size, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto torch_time = measure_time_us([&]() {
        for (size_t i = 0; i < num_ops; ++i) {
            size_t offset = i * stride;
            torch_dst.slice(0, offset, offset + slice_size) = torch_src;
        }
    },
                                      20);

    print_comparison("Worst Case: Fragmented Access (10 x 1K, stride 10K)", lfs_time, torch_time, data_mb);
}