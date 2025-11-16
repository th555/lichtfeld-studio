/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/tensor.hpp"
#include "framebuffer_factory.hpp"
#include "gl_resources.hpp"
#include "shader.hpp"
#include "shader_manager.hpp"
#include <memory>

namespace lfs::rendering {
    // Import Tensor from lfs::core
    using lfs::core::Tensor;

    class ScreenQuadRenderer {
    protected:
        VAO quadVAO_;
        VBO quadVBO_;

    public:
        std::shared_ptr<FrameBuffer> framebuffer;

        explicit ScreenQuadRenderer(FrameBufferMode mode = FrameBufferMode::CPU);
        virtual ~ScreenQuadRenderer() = default;

        // Updated to return Result for consistency
        virtual Result<void> render(std::shared_ptr<Shader> shader) const;
        Result<void> render(ManagedShader& shader) const;

        virtual Result<void> uploadData(const unsigned char* image, int width_, int height_);
        Result<void> uploadFromCUDA(const Tensor& cuda_image, int width, int height);

        bool isInteropEnabled() const;

    protected:
        virtual GLuint getTextureID() const;
    };
} // namespace lfs::rendering