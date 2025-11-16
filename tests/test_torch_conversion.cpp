/* Test to_torch conversion */
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "core_new/tensor.hpp"

// Helper to convert lfs::core::Tensor to torch::Tensor
torch::Tensor to_torch(const lfs::core::Tensor& lfs_tensor) {
    auto vec = lfs_tensor.cpu().to_vector();
    std::vector<int64_t> torch_shape;
    for (size_t i = 0; i < lfs_tensor.ndim(); i++) {
        torch_shape.push_back(lfs_tensor.shape()[i]);
    }
    auto torch_t = torch::from_blob(vec.data(), torch_shape, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return lfs_tensor.device() == lfs::core::Device::CUDA ? torch_t.to(torch::kCUDA) : torch_t;
}

TEST(TorchConversionTest, BasicConversion) {
    auto lfs_t = lfs::core::Tensor::randn({10, 1}, lfs::core::Device::CUDA);

    // Convert multiple times
    auto torch_t1 = to_torch(lfs_t);
    auto torch_t2 = to_torch(lfs_t);

    EXPECT_TRUE(torch_t1.defined());
    EXPECT_TRUE(torch_t2.defined());
    EXPECT_TRUE(torch_t1.is_cuda());
    EXPECT_TRUE(torch_t2.is_cuda());

    // Try sigmoid
    auto sig1 = torch::sigmoid(torch_t1);
    auto sig2 = torch::sigmoid(torch_t2);

    EXPECT_TRUE(sig1.defined());
    EXPECT_TRUE(sig2.defined());
}
