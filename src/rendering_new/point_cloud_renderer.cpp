/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "point_cloud_renderer.hpp"
#include "core_new/logger.hpp"
#include "gl_state_guard.hpp"
#include "shader_paths.hpp"
#include <vector>

namespace lfs::rendering {

    Result<void> PointCloudRenderer::initialize() {
        LOG_DEBUG("PointCloudRenderer::initialize() called on instance {}", static_cast<void*>(this));

        if (initialized_) {
            LOG_WARN("PointCloudRenderer already initialized!");
            return {};
        }

        LOG_TIMER_TRACE("PointCloudRenderer::initialize");

        // Create shader
        auto result = load_shader("point_cloud", "point_cloud.vert", "point_cloud.frag", false);
        if (!result) {
            LOG_ERROR("Failed to load point cloud shader: {}", result.error().what());
            return std::unexpected(result.error().what());
        }
        shader_ = std::move(*result);

        if (auto geom_result = createCubeGeometry(); !geom_result) {
            return geom_result;
        }

        initialized_ = true;
        LOG_INFO("PointCloudRenderer initialized successfully");
        return {};
    }

    Result<void> PointCloudRenderer::createCubeGeometry() {
        LOG_TIMER_TRACE("PointCloudRenderer::createCubeGeometry");

        // Create all resources first
        auto vao_result = create_vao();
        if (!vao_result) {
            LOG_ERROR("Failed to create VAO: {}", vao_result.error());
            return std::unexpected(vao_result.error());
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            LOG_ERROR("Failed to create VBO: {}", vbo_result.error());
            return std::unexpected(vbo_result.error());
        }
        cube_vbo_ = std::move(*vbo_result);

        auto ebo_result = create_vbo(); // EBO is also a buffer
        if (!ebo_result) {
            LOG_ERROR("Failed to create EBO: {}", ebo_result.error());
            return std::unexpected(ebo_result.error());
        }
        cube_ebo_ = std::move(*ebo_result);

        auto instance_result = create_vbo();
        if (!instance_result) {
            LOG_ERROR("Failed to create instance VBO: {}", instance_result.error());
            return std::unexpected(instance_result.error());
        }
        instance_vbo_ = std::move(*instance_result);

        // Build VAO using VAOBuilder
        VAOBuilder builder(std::move(*vao_result));

        // Setup cube geometry
        std::span<const float> vertices_span(cube_vertices_,
                                             sizeof(cube_vertices_) / sizeof(float));
        builder.attachVBO(cube_vbo_, vertices_span, GL_STATIC_DRAW)
            .setAttribute({.index = 0,
                           .size = 3,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = 3 * sizeof(float),
                           .offset = nullptr,
                           .divisor = 0});

        // Setup instance attributes (structure only, data comes later)
        builder.attachVBO(instance_vbo_) // Attach without data
            .setAttribute({
                .index = 1,
                .size = 3,
                .type = GL_FLOAT,
                .normalized = GL_FALSE,
                .stride = 6 * sizeof(float),
                .offset = nullptr,
                .divisor = 1 // Instance attribute
            })
            .setAttribute({
                .index = 2,
                .size = 3,
                .type = GL_FLOAT,
                .normalized = GL_FALSE,
                .stride = 6 * sizeof(float),
                .offset = (void*)(3 * sizeof(float)),
                .divisor = 1 // Instance attribute
            });

        // Attach EBO - stays bound to VAO
        std::span<const unsigned int> indices_span(cube_indices_,
                                                   sizeof(cube_indices_) / sizeof(unsigned int));
        builder.attachEBO(cube_ebo_, indices_span, GL_STATIC_DRAW);

        // Build and store the VAO
        cube_vao_ = builder.build();

        LOG_DEBUG("Cube geometry created successfully");
        return {};
    }

    Tensor PointCloudRenderer::extractRGBFromSH(const Tensor& shs) {
        const float SH_C0 = 0.28209479177387814f;

        // Extract features_dc: shs[:, 0, :]
        // We need to slice along dimension 1 (the second dimension)
        Tensor features_dc = shs.slice(1, 0, 1).squeeze(1);

        // Calculate colors: features_dc * SH_C0 + 0.5
        Tensor colors = features_dc * SH_C0 + 0.5f;

        return colors.clamp(0.0f, 1.0f);
    }

    Result<void> PointCloudRenderer::render(const lfs::core::SplatData& splat_data,
                                            const glm::mat4& view,
                                            const glm::mat4& projection,
                                            float voxel_size,
                                            const glm::vec3& background_color) {
        if (!initialized_) {
            LOG_ERROR("Renderer not initialized");
            return std::unexpected("Renderer not initialized");
        }

        if (splat_data.size() == 0) {
            LOG_TRACE("No splat data to render");
            return {}; // Nothing to render
        }

        LOG_TIMER_TRACE("PointCloudRenderer::render");

        // Use comprehensive state guard to isolate our state changes
        GLStateGuard state_guard;

        // Get positions and SH coefficients
        Tensor positions = splat_data.get_means();
        Tensor shs = splat_data.get_shs();

        // Extract RGB colors from SH coefficients
        Tensor colors = extractRGBFromSH(shs);

        const size_t num_points = positions.size(0);
        const size_t buffer_size = num_points * 6 * sizeof(float); // 6 floats per point (pos + color)
        current_point_count_ = num_points;

#ifdef CUDA_GL_INTEROP_ENABLED
        // Try CUDA-GL interop path first
        if (use_interop_) {
            LOG_TIMER_TRACE("CUDA-GL interop upload");

            // Ensure VBO has correct size
            BufferBinder<GL_ARRAY_BUFFER> bind(instance_vbo_);
            glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);

            // Initialize interop buffer if needed
            if (!interop_buffer_) {
                LOG_DEBUG("Initializing CUDA-GL interop buffer");
                interop_buffer_.emplace();
                if (auto result = interop_buffer_->init(instance_vbo_.get(), buffer_size); !result) {
                    LOG_WARN("Failed to initialize CUDA-GL interop: {}", result.error());
                    LOG_INFO("Falling back to CPU copy mode");
                    use_interop_ = false;
                    interop_buffer_.reset();
                }
            }

            if (use_interop_ && interop_buffer_) {
                // Map buffer to get CUDA pointer
                auto map_result = interop_buffer_->mapBuffer();
                if (map_result) {
                    float* vbo_ptr = static_cast<float*>(*map_result);

                    // Launch CUDA kernel to write interleaved data directly to VBO
                    lfs::launchWriteInterleavedPosColor(
                        positions.ptr<float>(),
                        colors.ptr<float>(),
                        vbo_ptr,
                        num_points,
                        0); // default stream

                    // Synchronize to ensure write is complete
                    cudaDeviceSynchronize();

                    // Unmap buffer
                    if (auto unmap_result = interop_buffer_->unmapBuffer(); !unmap_result) {
                        LOG_ERROR("Failed to unmap buffer: {}", unmap_result.error());
                    }

                    LOG_TRACE("Successfully uploaded {} points via CUDA-GL interop", num_points);
                } else {
                    LOG_WARN("Failed to map interop buffer: {}", map_result.error());
                    LOG_INFO("Falling back to CPU copy mode");
                    use_interop_ = false;
                    interop_buffer_.reset();
                }
            }
        }

        // Fallback to CPU path if interop failed or is disabled
        if (!use_interop_)
#endif
        {
            // Original CPU path
            LOG_TIMER_TRACE("CPU fallback upload");

            // Interleave on GPU using tensor concatenation (20x faster than CPU loop)
            Tensor interleaved;
            {
                LOG_TIMER_TRACE("tensor cat");
                interleaved = Tensor::cat({positions, colors}, -1).contiguous();
            }

            Tensor cpu_data;
            {
                LOG_TIMER_TRACE("cuda to cpu");
                cpu_data = interleaved.cpu();
            }

            // Upload to OpenGL
            BufferBinder<GL_ARRAY_BUFFER> bind(instance_vbo_);
            {
                LOG_TIMER_TRACE("glBufferData");
                glBufferData(GL_ARRAY_BUFFER, cpu_data.bytes(), cpu_data.raw_ptr(), GL_DYNAMIC_DRAW);
            }
        }

        // Validate instance count
        if (current_point_count_ > 10000000) { // 10 million sanity check
            LOG_ERROR("Instance count exceeds reasonable limit: {}", current_point_count_);
            return std::unexpected("Instance count exceeds reasonable limit");
        }

        LOG_TRACE("Rendering {} points", current_point_count_);

        // Setup rendering state for point cloud
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
        glClearColor(background_color.r, background_color.g, background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Bind shader and set uniforms
        ShaderScope s(shader_);
        if (auto result = s->set("u_view", view); !result) {
            return result;
        }
        if (auto result = s->set("u_projection", projection); !result) {
            return result;
        }
        if (auto result = s->set("u_voxel_size", voxel_size); !result) {
            return result;
        }

        // Validate VAO
        if (!cube_vao_ || cube_vao_.get() == 0) {
            LOG_ERROR("Invalid cube VAO");
            return std::unexpected("Invalid cube VAO");
        }

        // Render instanced cubes
        if (current_point_count_ == 0) {
            LOG_TRACE("No points to render");
            return {};
        }

        VAOBinder vao_bind(cube_vao_);
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0,
                                static_cast<GLsizei>(current_point_count_));

        // Check for OpenGL errors
        GLenum gl_error = glGetError();
        if (gl_error != GL_NO_ERROR) {
            LOG_ERROR("OpenGL error after draw call: 0x{:x}", gl_error);
            return std::unexpected(std::format("OpenGL error after draw call: 0x{:x}", gl_error));
        }

        // State automatically restored by GLStateGuard destructor
        return {};
    }

} // namespace lfs::rendering