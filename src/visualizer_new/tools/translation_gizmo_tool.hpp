/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering_new/rendering.hpp"
#include "tool_base.hpp"
#include <glm/glm.hpp>
#include <memory>

namespace lfs::vis::tools {

    class TranslationGizmoTool : public ToolBase {
    public:
        TranslationGizmoTool();
        ~TranslationGizmoTool() override = default;

        std::string_view getName() const override { return "Translation Gizmo"; }
        std::string_view getDescription() const override {
            return "Interactive 3D gizmo for translating objects in world space";
        }

        bool initialize(const ToolContext& ctx) override;
        void shutdown() override;
        void update(const ToolContext& ctx) override;
        void renderUI(const lfs::vis::gui::UIContext& ui_ctx, bool* p_open) override;

        // Gizmo-specific methods
        bool isInteracting() const { return is_dragging_; }
        bool handleMouseButton(int button, int action, double x, double y, const ToolContext& ctx);
        bool handleMouseMove(double x, double y, const ToolContext& ctx);

        // Get the current transform
        lfs::geometry::EuclideanTransform getTransform() const { return current_transform_; }

    protected:
        void onEnabledChanged(bool enabled) override;

    private:
        // Gizmo state
        lfs::geometry::EuclideanTransform current_transform_;
        std::shared_ptr<lfs::rendering::GizmoInteraction> gizmo_interaction_;

        // Interaction state
        bool is_dragging_ = false;
        lfs::rendering::GizmoElement hovered_element_ = lfs::rendering::GizmoElement::None;
        lfs::rendering::GizmoElement selected_element_ = lfs::rendering::GizmoElement::None;

        // Drag state
        glm::vec3 drag_start_position_;
        glm::vec3 drag_start_gizmo_pos_;
        lfs::geometry::EuclideanTransform drag_start_transform_;

        // Settings
        float gizmo_scale_ = 1.0f;
        bool show_in_viewport_ = true;
        bool apply_to_world_ = true; // Apply to world transform vs object transform

        // Store context for UI updates
        const ToolContext* tool_context_ = nullptr;

        // Helper methods
        glm::mat4 getViewMatrix(const ToolContext& ctx) const;
        glm::mat4 getProjectionMatrix(const ToolContext& ctx) const;
        void updateWorldTransform(const ToolContext& ctx);
    };

} // namespace lfs::vis::tools