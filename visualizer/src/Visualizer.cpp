// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#include <igl/file_dialog_open.h>
#include <solvers/materials/Material.h>
#include <solvers/materials/Rve.h>
#include <utilities/runtime/NeonLog.h>
#include <utility>
#include <visualizer/Visualizer.h>

visualizer::Visualizer::Visualizer() {
    viewer_.plugins.push_back(&menu_);
    menu_.callback_draw_viewer_menu = [&]() { GeneratorMenu(); };
}

visualizer::Visualizer::Visualizer(std::shared_ptr<meshing::Mesh> mesh) : mesh_(std::move(mesh)) {
    // Add the menu plugin to the viewer so it shows up.
    viewer_.plugins.push_back(&menu_);

    menu_.callback_draw_viewer_menu = [&]() { GeneratorMenu(); };

    viewer_.data().set_mesh(mesh_->RenderablePositions(), mesh_->faces);
}

auto visualizer::Visualizer::Launch() -> void { viewer_.launch(); }
auto visualizer::Visualizer::AddObjectToViewer() -> void {}
auto visualizer::Visualizer::Refresh() -> void {
    if (!(viewer_.data().F.rows() == 0 && viewer_.data().V.rows() == 0)) { viewer_.data().clear(); }

    viewer_.data().set_mesh(mesh_->RenderablePositions(), mesh_->faces);
}
auto visualizer::Visualizer::UpdateVertexPositions(const VectorXr &displacements) -> void {}
auto visualizer::Visualizer::GeneratorMenu() -> void {
    if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        float p = ImGui::GetStyle().FramePadding.x;
        if (ImGui::Button("Load##Mesh", ImVec2((w - p) / 2.f, 0))) {
            const std::string f_name = igl::file_dialog_open();
            NEON_ASSERT_WARN(f_name.length() > 0, "File name empty.");
            const auto file_type = meshing::ReadFileExtension(f_name);
            mesh_->ReloadMesh(f_name, file_type);
            Refresh();
        }
        ImGui::SameLine(0, p);
        if (ImGui::Button("Save##Mesh", ImVec2((w - p) / 2.f, 0))) { viewer_.open_dialog_save_mesh(); }
    }

    // Viewing options
    if (ImGui::CollapsingHeader("Viewing Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Center object", ImVec2(-1, 0))) {
            viewer_.core().align_camera_center(viewer_.data().V, viewer_.data().F);
        }
        if (ImGui::Button("Snap canonical view", ImVec2(-1, 0))) { viewer_.snap_to_canonical_quaternion(); }

        // Zoom
        ImGui::PushItemWidth(80 * menu_.menu_scaling());
        ImGui::DragFloat("Zoom", &(viewer_.core().camera_zoom), 0.05f, 0.1f, 20.0f);

        // Select rotation type
        int rotation_type = static_cast<int>(viewer_.core().rotation_type);
        static Eigen::Quaternionf trackball_angle = Eigen::Quaternionf::Identity();
        static bool orthographic = true;
        if (ImGui::Combo("Camera Type", &rotation_type, "Trackball\0Two Axes\0002D Mode\0\0")) {
            using RT = igl::opengl::ViewerCore::RotationType;
            auto new_type = static_cast<RT>(rotation_type);
            if (new_type != viewer_.core().rotation_type) {
                if (new_type == RT::ROTATION_TYPE_NO_ROTATION) {
                    trackball_angle = viewer_.core().trackball_angle;
                    orthographic = viewer_.core().orthographic;
                    viewer_.core().trackball_angle = Eigen::Quaternionf::Identity();
                    viewer_.core().orthographic = true;
                } else if (viewer_.core().rotation_type == RT::ROTATION_TYPE_NO_ROTATION) {
                    viewer_.core().trackball_angle = trackball_angle;
                    viewer_.core().orthographic = orthographic;
                }
                viewer_.core().set_rotation_type(new_type);
            }
        }

        // Orthographic view
        ImGui::Checkbox("Orthographic view", &(viewer_.core().orthographic));
        ImGui::PopItemWidth();
    }

    // Helper for setting viewport specific mesh options
    auto make_checkbox = [&](const char *label, unsigned int &option) {
        return ImGui::Checkbox(
                label, [&]() { return viewer_.core().is_set(option); },
                [&](bool value) { return viewer_.core().set(option, value); });
    };

    // Draw options
    if (ImGui::CollapsingHeader("Draw Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Checkbox("Face-based", &(viewer_.data().face_based))) {
            viewer_.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
        }
        make_checkbox("Show texture", viewer_.data().show_texture);
        if (ImGui::Checkbox("Invert normals", &(viewer_.data().invert_normals))) {
            viewer_.data().dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
        }
        make_checkbox("Show overlay", viewer_.data().show_overlay);
        make_checkbox("Show overlay depth", viewer_.data().show_overlay_depth);
        ImGui::ColorEdit4("Background", viewer_.core().background_color.data(),
                          ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
        ImGui::ColorEdit4("Line color", viewer_.data().line_color.data(),
                          ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
        ImGui::DragFloat("Shininess", &(viewer_.data().shininess), 0.05f, 0.0f, 100.0f);
        ImGui::PopItemWidth();
    }

    // Overlays
    if (ImGui::CollapsingHeader("Overlays", ImGuiTreeNodeFlags_DefaultOpen)) {
        make_checkbox("Wireframe", viewer_.data().show_lines);
        make_checkbox("Fill", viewer_.data().show_faces);
        make_checkbox("Show vertex labels", viewer_.data().show_vertex_labels);
        make_checkbox("Show faces labels", viewer_.data().show_face_labels);
        make_checkbox("Show extra labels", viewer_.data().show_custom_labels);
    }

    // Shape Generator
    if (ImGui::CollapsingHeader("Shape Generator", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        float p = ImGui::GetStyle().FramePadding.x;
        ImGui::InputInt("Rve Dim", &rve_dims_);
        ImGui::InputInt("Void Dim", &void_dims_);
        ImGui::InputInt("N Voids", &n_voids_);
        if (ImGui::Button("Generate##Shape Generator", ImVec2((w - p) / 2.f, 0))) {
            NEON_LOG_INFO("Generating Shape");
            rve_ = std::make_unique<solvers::materials::Rve>(
                    Vector3i(rve_dims_, rve_dims_, rve_dims_),
                    solvers::materials::MaterialFromEandv(1, "m_1", 1000, 0.3));
            MatrixXr V;
            MatrixXi F;
            if (mesh_ == nullptr) {
                NEON_LOG_INFO("Computing new grid mesh...");
                rve_->ComputeGridMesh(Vector3i(void_dims_, void_dims_, void_dims_), n_voids_, true, V, F);
                mesh_ = std::make_shared<meshing::Mesh>(V, F, "Yzpq");
                Refresh();
            } else {
                NEON_LOG_INFO("Reloading grid mesh...");
                rve_->ComputeGridMesh(Vector3i(void_dims_, void_dims_, void_dims_), n_voids_, true, V, F);
                mesh_->ReloadMesh(V, F, "Yzpq");
                Refresh();
            }

            NEON_LOG_INFO("Regeneration complete");
        }
    }

    if (ImGui::CollapsingHeader("Homogenization", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        float p = ImGui::GetStyle().FramePadding.x;

        ImGui::InputDouble("Young's Modulus", &youngs_modulus_);
        ImGui::InputDouble("Poisson's Ratio", &poissons_ratio_);

        ImGui::LabelText("E", "%.0e", youngs_modulus_);
        ImGui::LabelText("v", "%.0e", poissons_ratio_);

        Real E_x = 0;
        Real E_y = 0;
        Real E_z = 0;
        Real G_x = 0;
        Real G_y = 0;
        Real G_z = 0;
        ImGui::LabelText("E_x", "%.0e", E_x);
        ImGui::LabelText("E_y", "%.0e", E_y);
        ImGui::LabelText("E_z", "%.0e", E_z);
        ImGui::LabelText("G_x", "%.0e", G_x);
        ImGui::LabelText("G_y", "%.0e", G_y);
        ImGui::LabelText("G_z", "%.0e", G_z);

        if (ImGui::Button("Homogenize##Homogenization", ImVec2((w - p) / 2.f, 0))) {
            NEON_LOG_INFO("Homogenizing current geometry...");
            if (mesh_ == nullptr) { NEON_LOG_ERROR("Cannot homogenize empty mesh, aborting operation!!"); }

            const auto homo_task = [&]() -> void {
                NEON_LOG_INFO("Background homogenization task started");
                rve_->Homogenize();
            };
            std::thread t(homo_task);
            t.join();
            
            const solvers::materials::Homogenization::MaterialCoefficients coeffs = rve_->Homogenized()->Coefficients();

            E_x = coeffs.E_11;
            E_y = coeffs.E_22;
            E_z = coeffs.E_33;
            G_x = coeffs.G_23;
            G_y = coeffs.G_31;
            G_z = coeffs.G_12;
        }
    }
}
