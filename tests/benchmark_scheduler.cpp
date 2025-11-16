/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include "optimizers/fused_adam.hpp"
#include "optimizers/scheduler.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace lfs::core;
using namespace lfs::training;

namespace {

// Helper to create test data
SplatData create_test_splat_data(size_t n_points = 100000) {
    auto means = Tensor::randn({n_points, 3}, Device::CUDA);
    auto sh0 = Tensor::randn({n_points, 1, 3}, Device::CUDA);
    auto shN = Tensor::randn({n_points, 15, 3}, Device::CUDA);
    auto scaling = Tensor::randn({n_points, 3}, Device::CUDA);
    auto rotation = Tensor::randn({n_points, 4}, Device::CUDA);
    auto opacity = Tensor::randn({n_points, 1}, Device::CUDA);

    SplatData splat_data(3, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    splat_data.allocate_gradients();
    return splat_data;
}

// Helper to create torch optimizer for comparison
std::unique_ptr<gs::training::FusedAdam> create_torch_optimizer(SplatData& splat_data) {
    // Convert to torch tensors for old optimizer
    std::vector<torch::Tensor> params;

    auto means_torch = torch::from_blob(splat_data.means().ptr<float>(),
                                       {static_cast<int64_t>(splat_data.means().shape()[0]), 3},
                                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto sh0_torch = torch::from_blob(splat_data.sh0().ptr<float>(),
                                     {static_cast<int64_t>(splat_data.sh0().shape()[0]), 1, 3},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto shN_torch = torch::from_blob(splat_data.shN().ptr<float>(),
                                     {static_cast<int64_t>(splat_data.shN().shape()[0]), 15, 3},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto scaling_torch = torch::from_blob(splat_data.scaling_raw().ptr<float>(),
                                         {static_cast<int64_t>(splat_data.scaling_raw().shape()[0]), 3},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto rotation_torch = torch::from_blob(splat_data.rotation_raw().ptr<float>(),
                                          {static_cast<int64_t>(splat_data.rotation_raw().shape()[0]), 4},
                                          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto opacity_torch = torch::from_blob(splat_data.opacity_raw().ptr<float>(),
                                         {static_cast<int64_t>(splat_data.opacity_raw().shape()[0]), 1},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    means_torch.set_requires_grad(true);
    sh0_torch.set_requires_grad(true);
    shN_torch.set_requires_grad(true);
    scaling_torch.set_requires_grad(true);
    rotation_torch.set_requires_grad(true);
    opacity_torch.set_requires_grad(true);

    params.push_back(means_torch);
    params.push_back(sh0_torch);
    params.push_back(shN_torch);
    params.push_back(scaling_torch);
    params.push_back(rotation_torch);
    params.push_back(opacity_torch);

    auto options = std::make_unique<gs::training::FusedAdam::Options>(0.001);
    options->betas(std::make_tuple(0.9, 0.999));
    options->eps(1e-8);

    return std::make_unique<gs::training::FusedAdam>(std::move(params), std::move(options));
}

// Timing utilities
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }

    double stop_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    double stop_us() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

void print_benchmark_header(const std::string& title) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ " << std::left << std::setw(61) << title << "║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
}

void print_result(const std::string& test_name, double old_time, double new_time, const std::string& unit = "μs") {
    double speedup = old_time / new_time;
    std::cout << std::left << std::setw(40) << test_name << " | ";
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << old_time << " " << unit << " | ";
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << new_time << " " << unit << " | ";
    std::cout << std::right << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x";

    if (speedup > 1.1) {
        std::cout << " ✓";
    } else if (speedup < 0.9) {
        std::cout << " ⚠";
    }
    std::cout << "\n";
}

// ===================================================================================
// Benchmark: Scheduler Construction
// ===================================================================================

TEST(SchedulerBenchmark, ConstructionOverhead_ExponentialLR) {
    print_benchmark_header("ExponentialLR Construction");
    std::cout << std::left << std::setw(40) << "Test" << " | ";
    std::cout << std::right << std::setw(10) << "Old (Torch)" << "    | ";
    std::cout << std::right << std::setw(10) << "New (Free)" << "    | ";
    std::cout << std::right << std::setw(6) << "Speedup\n";
    std::cout << std::string(80, '-') << "\n";

    const int n_trials = 1000;

    // Benchmark old torch-based scheduler
    auto splat_data = create_test_splat_data(100000);
    auto torch_opt = create_torch_optimizer(splat_data);

    Timer timer;
    timer.start();
    for (int i = 0; i < n_trials; i++) {
        gs::training::ExponentialLR old_scheduler(*torch_opt, 0.99);
    }
    double old_time = timer.stop_us() / n_trials;

    // Benchmark new libtorch-free scheduler
    AdamConfig config;
    config.lr = 0.001f;
    AdamOptimizer new_opt(splat_data, config);

    timer.start();
    for (int i = 0; i < n_trials; i++) {
        lfs::training::ExponentialLR new_scheduler(new_opt, 0.99);
    }
    double new_time = timer.stop_us() / n_trials;

    print_result("Construction (1000 trials avg)", old_time, new_time);
}

TEST(SchedulerBenchmark, ConstructionOverhead_WarmupExponentialLR) {
    const int n_trials = 1000;

    // Benchmark old torch-based scheduler
    auto splat_data = create_test_splat_data(100000);
    auto torch_opt = create_torch_optimizer(splat_data);

    Timer timer;
    timer.start();
    for (int i = 0; i < n_trials; i++) {
        gs::training::WarmupExponentialLR old_scheduler(*torch_opt, 0.99, 100, 0.1);
    }
    double old_time = timer.stop_us() / n_trials;

    // Benchmark new libtorch-free scheduler
    AdamConfig config;
    config.lr = 0.001f;
    AdamOptimizer new_opt(splat_data, config);

    timer.start();
    for (int i = 0; i < n_trials; i++) {
        lfs::training::WarmupExponentialLR new_scheduler(new_opt, 0.99, 100, 0.1);
    }
    double new_time = timer.stop_us() / n_trials;

    print_result("WarmupExpLR construction (1000 trials)", old_time, new_time);
}

// ===================================================================================
// Benchmark: Single Step Performance
// ===================================================================================

TEST(SchedulerBenchmark, SingleStep_ExponentialLR) {
    print_benchmark_header("ExponentialLR Single Step");
    std::cout << std::left << std::setw(40) << "Test" << " | ";
    std::cout << std::right << std::setw(10) << "Old (Torch)" << "    | ";
    std::cout << std::right << std::setw(10) << "New (Free)" << "    | ";
    std::cout << std::right << std::setw(6) << "Speedup\n";
    std::cout << std::string(80, '-') << "\n";

    const int n_trials = 10000;

    // Benchmark old torch-based scheduler
    auto splat_data = create_test_splat_data(100000);
    auto torch_opt = create_torch_optimizer(splat_data);
    gs::training::ExponentialLR old_scheduler(*torch_opt, 0.99);

    Timer timer;
    timer.start();
    for (int i = 0; i < n_trials; i++) {
        old_scheduler.step();
    }
    double old_time = timer.stop_us() / n_trials;

    // Benchmark new libtorch-free scheduler
    AdamConfig config;
    config.lr = 0.001f;
    AdamOptimizer new_opt(splat_data, config);
    lfs::training::ExponentialLR new_scheduler(new_opt, 0.99);

    timer.start();
    for (int i = 0; i < n_trials; i++) {
        new_scheduler.step();
    }
    double new_time = timer.stop_us() / n_trials;

    print_result("Single step (10k trials avg)", old_time, new_time);
}

TEST(SchedulerBenchmark, SingleStep_WarmupExponentialLR) {
    const int n_trials = 10000;

    // Benchmark old torch-based scheduler (warmup phase)
    auto splat_data = create_test_splat_data(100000);
    auto torch_opt = create_torch_optimizer(splat_data);
    gs::training::WarmupExponentialLR old_scheduler(*torch_opt, 0.99, 1000, 0.1);

    Timer timer;
    timer.start();
    for (int i = 0; i < n_trials; i++) {
        old_scheduler.step();
    }
    double old_time_warmup = timer.stop_us() / n_trials;

    // Benchmark new libtorch-free scheduler (warmup phase)
    AdamConfig config;
    config.lr = 0.001f;
    AdamOptimizer new_opt(splat_data, config);
    lfs::training::WarmupExponentialLR new_scheduler(new_opt, 0.99, 1000, 0.1);

    timer.start();
    for (int i = 0; i < n_trials; i++) {
        new_scheduler.step();
    }
    double new_time_warmup = timer.stop_us() / n_trials;

    print_result("Warmup phase step (10k trials)", old_time_warmup, new_time_warmup);

    // Benchmark decay phase
    auto splat_data2 = create_test_splat_data(100000);
    auto torch_opt2 = create_torch_optimizer(splat_data2);
    gs::training::WarmupExponentialLR old_scheduler2(*torch_opt2, 0.99, 100, 0.1);

    // Skip warmup
    for (int i = 0; i < 100; i++) {
        old_scheduler2.step();
    }

    timer.start();
    for (int i = 0; i < n_trials; i++) {
        old_scheduler2.step();
    }
    double old_time_decay = timer.stop_us() / n_trials;

    AdamConfig config2;
    config2.lr = 0.001f;
    AdamOptimizer new_opt2(splat_data2, config2);
    lfs::training::WarmupExponentialLR new_scheduler2(new_opt2, 0.99, 100, 0.1);

    // Skip warmup
    for (int i = 0; i < 100; i++) {
        new_scheduler2.step();
    }

    timer.start();
    for (int i = 0; i < n_trials; i++) {
        new_scheduler2.step();
    }
    double new_time_decay = timer.stop_us() / n_trials;

    print_result("Decay phase step (10k trials)", old_time_decay, new_time_decay);
}

// ===================================================================================
// Benchmark: Realistic Training Scenario
// ===================================================================================

TEST(SchedulerBenchmark, RealisticTraining_30kIterations) {
    print_benchmark_header("Realistic Gaussian Splatting Training (30k iterations)");
    std::cout << std::left << std::setw(40) << "Test" << " | ";
    std::cout << std::right << std::setw(10) << "Old (Torch)" << "    | ";
    std::cout << std::right << std::setw(10) << "New (Free)" << "    | ";
    std::cout << std::right << std::setw(6) << "Speedup\n";
    std::cout << std::string(80, '-') << "\n";

    const int n_iterations = 30000;
    const int warmup_steps = 100;
    const double gamma = std::pow(0.01, 1.0 / 30000.0);  // Decay to 1% over 30k steps

    // Benchmark old torch-based approach
    auto splat_data_old = create_test_splat_data(100000);
    auto torch_opt = create_torch_optimizer(splat_data_old);
    gs::training::WarmupExponentialLR old_scheduler(*torch_opt, gamma, warmup_steps, 0.01);

    Timer timer;
    timer.start();
    for (int i = 0; i < n_iterations; i++) {
        old_scheduler.step();
    }
    double old_time = timer.stop_ms();

    // Benchmark new libtorch-free approach
    auto splat_data_new = create_test_splat_data(100000);
    AdamConfig config;
    config.lr = 1.6e-4f;
    AdamOptimizer new_opt(splat_data_new, config);
    lfs::training::WarmupExponentialLR new_scheduler(new_opt, gamma, warmup_steps, 0.01);

    timer.start();
    for (int i = 0; i < n_iterations; i++) {
        new_scheduler.step();
    }
    double new_time = timer.stop_ms();

    print_result("30k iterations total", old_time, new_time, "ms");
    print_result("Per iteration (avg)", old_time * 1000.0 / n_iterations, new_time * 1000.0 / n_iterations);
}

// ===================================================================================
// Benchmark: Memory Usage
// ===================================================================================

TEST(SchedulerBenchmark, MemoryFootprint) {
    print_benchmark_header("Memory Footprint Comparison");
    std::cout << std::left << std::setw(40) << "Component" << " | ";
    std::cout << std::right << std::setw(15) << "Size (bytes)\n";
    std::cout << std::string(60, '-') << "\n";

    auto splat_data = create_test_splat_data(100000);
    auto torch_opt = create_torch_optimizer(splat_data);
    AdamConfig config;
    config.lr = 0.001f;
    AdamOptimizer new_opt(splat_data, config);

    // Note: These are just the scheduler objects, not the full optimizer state
    std::cout << std::left << std::setw(40) << "Old ExponentialLR" << " | ";
    std::cout << std::right << std::setw(15) << sizeof(gs::training::ExponentialLR) << "\n";

    std::cout << std::left << std::setw(40) << "New ExponentialLR" << " | ";
    std::cout << std::right << std::setw(15) << sizeof(lfs::training::ExponentialLR) << "\n";

    std::cout << std::left << std::setw(40) << "Old WarmupExponentialLR" << " | ";
    std::cout << std::right << std::setw(15) << sizeof(gs::training::WarmupExponentialLR) << "\n";

    std::cout << std::left << std::setw(40) << "New WarmupExponentialLR" << " | ";
    std::cout << std::right << std::setw(15) << sizeof(lfs::training::WarmupExponentialLR) << "\n";

    std::cout << "\nNote: Scheduler objects are lightweight - main memory is in optimizer state.\n";
}

// ===================================================================================
// Benchmark: Throughput Test
// ===================================================================================

TEST(SchedulerBenchmark, Throughput_StepsPerSecond) {
    print_benchmark_header("Scheduler Throughput (steps/second)");
    std::cout << std::left << std::setw(40) << "Scheduler Type" << " | ";
    std::cout << std::right << std::setw(15) << "Old (steps/s)" << " | ";
    std::cout << std::right << std::setw(15) << "New (steps/s)" << " | ";
    std::cout << std::right << std::setw(6) << "Speedup\n";
    std::cout << std::string(85, '-') << "\n";

    const double test_duration_ms = 100.0;  // Run for 100ms

    // ExponentialLR throughput
    {
        auto splat_data = create_test_splat_data(100000);
        auto torch_opt = create_torch_optimizer(splat_data);
        gs::training::ExponentialLR old_scheduler(*torch_opt, 0.99);

        Timer timer;
        timer.start();
        int old_count = 0;
        while (timer.stop_ms() < test_duration_ms) {
            old_scheduler.step();
            old_count++;
        }
        double old_elapsed = timer.stop_ms();
        double old_throughput = (old_count / old_elapsed) * 1000.0;

        AdamConfig config;
        config.lr = 0.001f;
        AdamOptimizer new_opt(splat_data, config);
        lfs::training::ExponentialLR new_scheduler(new_opt, 0.99);

        timer.start();
        int new_count = 0;
        while (timer.stop_ms() < test_duration_ms) {
            new_scheduler.step();
            new_count++;
        }
        double new_elapsed = timer.stop_ms();
        double new_throughput = (new_count / new_elapsed) * 1000.0;

        std::cout << std::left << std::setw(40) << "ExponentialLR" << " | ";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(0) << old_throughput << " | ";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(0) << new_throughput << " | ";
        std::cout << std::right << std::setw(6) << std::fixed << std::setprecision(2) << (new_throughput / old_throughput) << "x\n";
    }

    // WarmupExponentialLR throughput
    {
        auto splat_data = create_test_splat_data(100000);
        auto torch_opt = create_torch_optimizer(splat_data);
        gs::training::WarmupExponentialLR old_scheduler(*torch_opt, 0.99, 1000, 0.1);

        Timer timer;
        timer.start();
        int old_count = 0;
        while (timer.stop_ms() < test_duration_ms) {
            old_scheduler.step();
            old_count++;
        }
        double old_elapsed = timer.stop_ms();
        double old_throughput = (old_count / old_elapsed) * 1000.0;

        AdamConfig config;
        config.lr = 0.001f;
        AdamOptimizer new_opt(splat_data, config);
        lfs::training::WarmupExponentialLR new_scheduler(new_opt, 0.99, 1000, 0.1);

        timer.start();
        int new_count = 0;
        while (timer.stop_ms() < test_duration_ms) {
            new_scheduler.step();
            new_count++;
        }
        double new_elapsed = timer.stop_ms();
        double new_throughput = (new_count / new_elapsed) * 1000.0;

        std::cout << std::left << std::setw(40) << "WarmupExponentialLR" << " | ";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(0) << old_throughput << " | ";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(0) << new_throughput << " | ";
        std::cout << std::right << std::setw(6) << std::fixed << std::setprecision(2) << (new_throughput / old_throughput) << "x\n";
    }
}

} // anonymous namespace
