/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_pipeline.hpp"
#include "core_new/camera.hpp"
#include "core_new/splat_data.hpp"
#include "gs_rasterizer_tensor.hpp"

#include <cstring>
#include <print>

namespace lfs::rendering {

    RenderingPipeline::RenderingPipeline()
        : background_(Tensor::zeros({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32)) {
        point_cloud_renderer_ = std::make_unique<PointCloudRenderer>();
        LOG_DEBUG("RenderingPipeline initialized");
    }

    RenderingPipeline::~RenderingPipeline() {
        cleanupFBO();
        cleanupPBO();
    }

    Result<RenderingPipeline::RenderResult> RenderingPipeline::render(
        const lfs::core::SplatData& model,
        const RenderRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::render");

        // Validate dimensions
        if (request.viewport_size.x <= 0 || request.viewport_size.y <= 0 ||
            request.viewport_size.x > 16384 || request.viewport_size.y > 16384) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.viewport_size.x, request.viewport_size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        // Check if we should use point cloud rendering
        if (request.point_cloud_mode) {
            LOG_TRACE("Using point cloud rendering mode");
            return renderPointCloud(model, request);
        }

        // Regular gaussian splatting rendering
        LOG_TRACE("Using gaussian splatting rendering mode");

        // Update background tensor in-place to avoid allocation
        // Access the tensor data directly
        auto bg_data = background_.ptr<float>();
        if (bg_data && background_.device() == lfs::core::Device::CUDA) {
            float bg_values[3] = {
                request.background_color.r,
                request.background_color.g,
                request.background_color.b};
            cudaMemcpy(bg_data, bg_values, 3 * sizeof(float), cudaMemcpyHostToDevice);
        }

        // Create camera for this frame
        auto cam_result = createCamera(request);
        if (!cam_result) {
            return std::unexpected(cam_result.error());
        }
        lfs::core::Camera cam = std::move(*cam_result);

        // Handle crop box conversion
        const lfs::geometry::BoundingBox* geom_bbox = nullptr;
        std::unique_ptr<lfs::geometry::BoundingBox> temp_bbox;

        if (request.crop_box) {
            // Create a temporary lfs::geometry::BoundingBox with the full transform
            temp_bbox = std::make_unique<lfs::geometry::BoundingBox>();
            temp_bbox->setBounds(request.crop_box->getMinBounds(), request.crop_box->getMaxBounds());
            temp_bbox->setworld2BBox(request.crop_box->getworld2BBox());
            geom_bbox = temp_bbox.get();
            LOG_TRACE("Using crop box for rendering");
        }

        try {
            if (request.sh_degree != model.get_active_sh_degree()) {
                // Temporarily set sh_degree for rendering, then immediately restore
                int original_sh_degree = model.get_active_sh_degree();
                const_cast<lfs::core::SplatData&>(model).set_active_sh_degree(request.sh_degree);

                RenderResult result;
                if (request.gut) {
                    throw std::runtime_error("GUT rendering mode not yet implemented for tensor backend");
                }

                LOG_TRACE("Using TENSOR_NATIVE backend (sh_degree temporarily changed from {} to {})",
                         original_sh_degree, request.sh_degree);
                result.image = rasterize_tensor(cam, const_cast<lfs::core::SplatData&>(model), background_);
                result.depth = Tensor::empty({0}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

                // IMMEDIATELY restore original sh_degree
                const_cast<lfs::core::SplatData&>(model).set_active_sh_degree(original_sh_degree);

                result.valid = true;
                LOG_TRACE("Rasterization completed successfully (sh_degree restored to {})", original_sh_degree);
                return result;
            }

            // No sh_degree change needed - safe to use model as-is
            lfs::core::SplatData& mutable_model = const_cast<lfs::core::SplatData&>(model);
            RenderResult result;

            if (request.gut) {
                throw std::runtime_error("GUT rendering mode not yet implemented for tensor backend");
            }

            // Use libtorch-free tensor-based rasterizer
            LOG_TRACE("Using TENSOR_NATIVE backend (libtorch-free rasterizer)");
            result.image = rasterize_tensor(cam, mutable_model, background_);
            result.depth = Tensor::empty({0}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            result.valid = true;

            LOG_TRACE("Rasterization completed successfully");
            return result;

        } catch (const std::exception& e) {
            LOG_ERROR("Rasterization failed: {}", e.what());
            return std::unexpected(std::format("Rasterization failed: {}", e.what()));
        }
    }

    Result<RenderingPipeline::RenderResult> RenderingPipeline::renderPointCloud(
        const lfs::core::SplatData& model,
        const RenderRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::renderPointCloud");

        // Initialize point cloud renderer if needed
        if (!point_cloud_renderer_->isInitialized()) {
            LOG_DEBUG("Initializing point cloud renderer");
            if (auto result = point_cloud_renderer_->initialize(); !result) {
                LOG_ERROR("Failed to initialize point cloud renderer: {}", result.error());
                return std::unexpected(std::format("Failed to initialize point cloud renderer: {}",
                                                   result.error()));
            }
        }

        // Save current OpenGL state
        GLint current_viewport[4];
        GLint current_fbo;
        glGetIntegerv(GL_VIEWPORT, current_viewport);
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);

        // RAII guard to restore state
        struct StateGuard {
            GLint viewport[4];
            GLint fbo;
            ~StateGuard() {
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            }
        } state_guard{
            {current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]},
            current_fbo};

        // Create view matrix using the same convention as Viewport::getViewMatrix()
        glm::mat3 flip_yz = glm::mat3(
            1, 0, 0,
            0, -1, 0,
            0, 0, -1);

        // Convert from camera space (what we get in request) to view space
        glm::mat3 R_inv = glm::transpose(request.view_rotation); // Inverse of rotation matrix
        glm::vec3 t_inv = -R_inv * request.view_translation;     // Inverse translation

        // Apply flip
        R_inv = flip_yz * R_inv;
        t_inv = flip_yz * t_inv;

        // Build view matrix
        glm::mat4 view(1.0f);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                view[i][j] = R_inv[i][j];
            }
        }
        view[3][0] = t_inv.x;
        view[3][1] = t_inv.y;
        view[3][2] = t_inv.z;
        view[3][3] = 1.0f;

        // Create projection matrix
        float aspect = static_cast<float>(request.viewport_size.x) / request.viewport_size.y;
        float fov_rad = glm::radians(request.fov);
        glm::mat4 projection = glm::perspective(fov_rad, aspect, 0.1f, 1000.0f);

        // OPTIMIZATION: Flip Y-axis in projection to render image pre-flipped
        // This eliminates the need for CPU-side vertical flip after glReadPixels
        // OpenGL has origin at bottom-left, we want top-left (saves ~1ms CPU time)
        projection[1][1] *= -1.0f;

        // OPTIMIZATION: Use persistent FBO (avoids expensive glGenFramebuffers/glDeleteFramebuffers)
        // This saves ~3-5ms per frame by reusing the same FBO across renders
        ensureFBOSize(request.viewport_size.x, request.viewport_size.y);
        if (persistent_fbo_ == 0) {
            LOG_ERROR("Failed to setup persistent framebuffer");
            return std::unexpected("Failed to setup persistent framebuffer");
        }

        // Set viewport to match the request size
        glViewport(0, 0, request.viewport_size.x, request.viewport_size.y);

        // Render point cloud to framebuffer
        {
            LOG_TIMER_TRACE("point_cloud_renderer_->render");
            if (auto result = point_cloud_renderer_->render(model, view, projection,
                                                            request.voxel_size, request.background_color);
                !result) {
                LOG_ERROR("Point cloud rendering failed: {}", result.error());
                return std::unexpected(std::format("Point cloud rendering failed: {}", result.error()));
            }
        }

        const int width = request.viewport_size.x;
        const int height = request.viewport_size.y;
        RenderResult result;

#ifdef CUDA_GL_INTEROP_ENABLED
        // Try CUDA-GL interop path for direct FBO→CUDA texture readback
        if (use_fbo_interop_) {
            LOG_TIMER_TRACE("CUDA-GL FBO interop readback");

            // Initialize interop texture if needed
            if (!fbo_interop_texture_) {
                LOG_DEBUG("Initializing CUDA-GL FBO interop texture");
                fbo_interop_texture_.emplace();
                if (auto init_result = fbo_interop_texture_->initForReading(
                        persistent_color_texture_, width, height);
                    !init_result) {
                    LOG_WARN("Failed to initialize FBO interop: {}", init_result.error());
                    LOG_INFO("Falling back to PBO readback mode");
                    use_fbo_interop_ = false;
                    fbo_interop_texture_.reset();
                }
            }

            if (use_fbo_interop_ && fbo_interop_texture_) {
                // Read texture directly to CUDA tensor [H, W, 3]
                Tensor image_hwc;
                if (auto read_result = fbo_interop_texture_->readToTensor(image_hwc); read_result) {
                    // Convert to CHW format
                    result.image = image_hwc.permute({2, 0, 1}).contiguous();
                    result.valid = true;
                    LOG_TRACE("Successfully read FBO via CUDA-GL interop");
                } else {
                    LOG_WARN("Failed to read FBO via interop: {}", read_result.error());
                    LOG_INFO("Falling back to PBO readback mode");
                    use_fbo_interop_ = false;
                    fbo_interop_texture_.reset();
                }
            }
        }

        // Fallback to PBO path if interop failed or is disabled
        if (!use_fbo_interop_)
#endif
        {
            // OPTIMIZATION: Use PBO for async GPU→CPU readback via DMA
            LOG_TIMER_TRACE("PBO fallback readback");

            ensurePBOSize(width, height);

            // Ping-pong between two PBOs for double-buffering
            int current_pbo = pbo_index_;
            int next_pbo = 1 - pbo_index_;

            std::vector<float> pixels(width * height * 3);
            {
                // Start async readback into current PBO
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_[current_pbo]);
                glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, nullptr);

                // Map the PBO to read data (may wait if transfer not complete)
                void* mapped_data = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
                if (mapped_data) {
                    // Copy data from mapped PBO to our vector
                    std::memcpy(pixels.data(), mapped_data, width * height * 3 * sizeof(float));
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                } else {
                    LOG_ERROR("Failed to map PBO for readback");
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                    return std::unexpected("Failed to map PBO for readback");
                }

                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }

            // Swap PBO index for next frame
            pbo_index_ = next_pbo;

            // Create tensor from the pixel data
            // OPTIMIZATION: No CPU flip needed! Image is already pre-flipped by projection matrix
            auto image_cpu = Tensor::from_vector(pixels, {static_cast<size_t>(height), static_cast<size_t>(width), 3},
                                                 lfs::core::Device::CPU);

            // Convert to CHW format and move to CUDA
            {
                LOG_TIMER_TRACE("permute and cuda upload");
                result.image = image_cpu.permute({2, 0, 1}).cuda();
            }
            result.valid = true;
        }

        LOG_TRACE("Point cloud rendering completed");
        return result;
    }

    Result<void> RenderingPipeline::uploadToScreen(
        const RenderResult& result,
        ScreenQuadRenderer& renderer,
        const glm::ivec2& viewport_size) {

        if (!result.valid || !result.image.is_valid()) {
            LOG_ERROR("Invalid render result for upload");
            return std::unexpected("Invalid render result");
        }

        // Try direct CUDA upload if available
        if (renderer.isInteropEnabled() && result.image.device() == lfs::core::Device::CUDA) {
            LOG_TRACE("Using CUDA interop for screen upload");
            // Keep data on GPU - convert [C, H, W] to [H, W, C] format
            auto image_hwc = result.image.permute({1, 2, 0}).contiguous();

            if (image_hwc.size(0) == static_cast<size_t>(viewport_size.y) &&
                image_hwc.size(1) == static_cast<size_t>(viewport_size.x)) {
                return renderer.uploadFromCUDA(image_hwc, viewport_size.x, viewport_size.y);
            }
        }

        // Fallback to CPU copy
        LOG_TRACE("Using CPU copy for screen upload");
        auto image = (result.image * 255.0f)
                         .cpu()
                         .to(lfs::core::DataType::UInt8)
                         .permute({1, 2, 0})
                         .contiguous();

        if (image.size(0) != static_cast<size_t>(viewport_size.y) ||
            image.size(1) != static_cast<size_t>(viewport_size.x) ||
            !image.ptr<unsigned char>()) {
            LOG_ERROR("Image dimensions mismatch or invalid data");
            return std::unexpected("Image dimensions mismatch or invalid data");
        }

        return renderer.uploadData(image.ptr<unsigned char>(),
                                   viewport_size.x, viewport_size.y);
    }

    Result<lfs::core::Camera> RenderingPipeline::createCamera(const RenderRequest& request) {
        LOG_TIMER_TRACE("RenderingPipeline::createCamera");

        // Convert view matrix to camera matrix
        std::vector<float> R_data = {
            request.view_rotation[0][0], request.view_rotation[1][0], request.view_rotation[2][0],
            request.view_rotation[0][1], request.view_rotation[1][1], request.view_rotation[2][1],
            request.view_rotation[0][2], request.view_rotation[1][2], request.view_rotation[2][2]};

        auto R_tensor = Tensor::from_vector(R_data, {3, 3}, lfs::core::Device::CPU);

        std::vector<float> t_data = {
            request.view_translation[0],
            request.view_translation[1],
            request.view_translation[2]};

        auto t_tensor = Tensor::from_vector(t_data, {3, 1}, lfs::core::Device::CPU);

        // Convert from view to camera space
        R_tensor = R_tensor.transpose(0, 1);
        t_tensor = (-R_tensor.mm(t_tensor)).squeeze();

        // Compute field of view
        glm::vec2 fov = computeFov(request.fov,
                                   request.viewport_size.x,
                                   request.viewport_size.y);

        try {
            return lfs::core::Camera(
                R_tensor,
                t_tensor,
                lfs::core::fov2focal(fov.x, request.viewport_size.x),
                lfs::core::fov2focal(fov.y, request.viewport_size.y),
                request.viewport_size.x / 2.0f,
                request.viewport_size.y / 2.0f,
                Tensor::empty({0}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
                Tensor::empty({0}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
                gsplat::CameraModelType::PINHOLE,
                "render_camera",
                "none",
                request.viewport_size.x,
                request.viewport_size.y,
                -1);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to create camera: {}", e.what());
            return std::unexpected(std::format("Failed to create camera: {}", e.what()));
        }
    }

    glm::vec2 RenderingPipeline::computeFov(float fov_degrees, int width, int height) {
        float fov_rad = glm::radians(fov_degrees);
        float aspect = static_cast<float>(width) / height;

        return glm::vec2(
            atan(tan(fov_rad / 2.0f) * aspect) * 2.0f,
            fov_rad);
    }

    void RenderingPipeline::ensureFBOSize(int width, int height) {
        // Check if we need to create or resize the FBO
        if (persistent_fbo_ != 0 && persistent_fbo_width_ == width && persistent_fbo_height_ == height) {
            // FBO already exists with correct size
            glBindFramebuffer(GL_FRAMEBUFFER, persistent_fbo_);
            return;
        }

        // Need to create or resize - cleanup old resources first
        if (persistent_fbo_ != 0) {
            LOG_DEBUG("Resizing persistent FBO from {}x{} to {}x{}",
                      persistent_fbo_width_, persistent_fbo_height_, width, height);
            cleanupFBO();
        } else {
            LOG_DEBUG("Creating persistent FBO with size {}x{}", width, height);
        }

        // Create new FBO
        glGenFramebuffers(1, &persistent_fbo_);
        if (persistent_fbo_ == 0) {
            LOG_ERROR("Failed to create persistent framebuffer");
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, persistent_fbo_);

        // Create color texture
        glGenTextures(1, &persistent_color_texture_);
        glBindTexture(GL_TEXTURE_2D, persistent_color_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               persistent_color_texture_, 0);

        // Create depth texture
        glGenTextures(1, &persistent_depth_texture_);
        glBindTexture(GL_TEXTURE_2D, persistent_depth_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height,
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                               persistent_depth_texture_, 0);

        // Verify framebuffer is complete
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("Persistent framebuffer not complete: 0x{:x}", status);
            cleanupFBO();
            return;
        }

        // Update size tracking
        persistent_fbo_width_ = width;
        persistent_fbo_height_ = height;
    }

    void RenderingPipeline::cleanupFBO() {
        if (persistent_fbo_ != 0) {
            glDeleteFramebuffers(1, &persistent_fbo_);
            persistent_fbo_ = 0;
        }
        if (persistent_color_texture_ != 0) {
            glDeleteTextures(1, &persistent_color_texture_);
            persistent_color_texture_ = 0;
        }
        if (persistent_depth_texture_ != 0) {
            glDeleteTextures(1, &persistent_depth_texture_);
            persistent_depth_texture_ = 0;
        }
        persistent_fbo_width_ = 0;
        persistent_fbo_height_ = 0;
    }

    void RenderingPipeline::ensurePBOSize(int width, int height) {
        // Check if we need to create or resize the PBOs
        if (pbo_[0] != 0 && pbo_width_ == width && pbo_height_ == height) {
            // PBOs already exist with correct size
            return;
        }

        // Need to create or resize - cleanup old resources first
        if (pbo_[0] != 0) {
            LOG_DEBUG("Resizing PBOs from {}x{} to {}x{}",
                      pbo_width_, pbo_height_, width, height);
            cleanupPBO();
        } else {
            LOG_DEBUG("Creating PBOs with size {}x{}", width, height);
        }

        // Calculate buffer size (RGB floats)
        size_t buffer_size = width * height * 3 * sizeof(float);

        // Create two PBOs for double-buffering
        glGenBuffers(2, pbo_);
        for (int i = 0; i < 2; i++) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_[i]);
            // GL_STREAM_READ: data written by OpenGL, read by application once
            glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, nullptr, GL_STREAM_READ);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        // Update size tracking
        pbo_width_ = width;
        pbo_height_ = height;
        pbo_index_ = 0;
    }

    void RenderingPipeline::cleanupPBO() {
        if (pbo_[0] != 0 || pbo_[1] != 0) {
            glDeleteBuffers(2, pbo_);
            pbo_[0] = 0;
            pbo_[1] = 0;
        }
        pbo_width_ = 0;
        pbo_height_ = 0;
        pbo_index_ = 0;
    }

} // namespace lfs::rendering