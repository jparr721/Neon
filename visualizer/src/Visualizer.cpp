// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#include <future>
#include <igl/file_dialog_open.h>
#include <solvers/materials/Material.h>
#include <solvers/materials/Rve.h>
#include <thread>
#include <utilities/filesystem/CsvFile.h>
#include <utilities/math/Time.h>
#include <utility>
#include <visualizer/Visualizer.h>

visualizer::Visualizer::Visualizer() { SetupMenus(); }

visualizer::Visualizer::Visualizer(std::shared_ptr<meshing::Mesh> mesh) : mesh_(std::move(mesh)) {
    SetupMenus();
    viewer_.data().set_mesh(mesh_->RenderablePositions(), mesh_->faces);
}

auto visualizer::Visualizer::Launch() -> void { viewer_.launch(); }
auto visualizer::Visualizer::AddObjectToViewer() -> void {}
auto visualizer::Visualizer::Refresh() -> void {
    if (!(viewer_.data().F.rows() == 0 && viewer_.data().V.rows() == 0)) { viewer_.data().clear(); }

    viewer_.data().set_mesh(mesh_->RenderablePositions(), mesh_->faces);
}
auto visualizer::Visualizer::UpdateVertexPositions(const VectorXr &displacements) -> void {}

auto visualizer::Visualizer::GenerateShape() -> void {
    NEON_LOG_INFO("Generating Shape");
    rve_ = std::make_unique<solvers::materials::Rve>(Vector3i(rve_dims_, rve_dims_, rve_dims_),
                                                     solvers::materials::MaterialFromEandv(1, "m_1", 1000, 0.3));
    MatrixXr V;
    MatrixXi F;
    meshing::ImplicitSurfaceGenerator<Real>::Inclusion inclusion{n_voids_, void_dims_, void_dims_, void_dims_};
    if (mesh_ == nullptr) {
        NEON_LOG_INFO("Computing new grid mesh...");
        if (n_voids_ == 0) {
            rve_->ComputeUniformMesh(V, F);
        } else {
            rve_->ComputeCompositeMesh(inclusion, isotropic_, V, F);
        }
        if (tetrahedralize_) {
            mesh_ = std::make_shared<meshing::Mesh>(V, F, tetgen_flags_);
        } else {
            mesh_ = std::make_shared<meshing::Mesh>(V, F);
        }
        Refresh();
    } else {
        NEON_LOG_INFO("Reloading grid mesh...");
        if (n_voids_ == 0) {
            rve_->ComputeUniformMesh(V, F);
        } else {
            rve_->ComputeCompositeMesh(inclusion, isotropic_, V, F);
        }
        if (tetrahedralize_) {
            mesh_->ReloadMesh(V, F, tetgen_flags_);
        } else {
            mesh_->ReloadMesh(V, F);
        }
        Refresh();
    }

    NEON_LOG_INFO("Regeneration complete");
}

auto visualizer::Visualizer::HomogenizeCurrentGeometry() -> void {
    NEON_LOG_INFO("Homogenizing current geometry...");
    if (mesh_ == nullptr) {
        NEON_LOG_ERROR("Cannot homogenize empty mesh, aborting operation!!");
    } else {
        rve_->Homogenize();

        const solvers::materials::Homogenization::MaterialCoefficients coeffs = rve_->Homogenized()->Coefficients();

        E_x = coeffs.E_11;
        E_y = coeffs.E_22;
        E_z = coeffs.E_33;
        G_x = coeffs.G_23;
        G_y = coeffs.G_31;
        G_z = coeffs.G_12;
        v_21 = coeffs.v_21;
        v_31 = coeffs.v_31;
        v_12 = coeffs.v_12;
        v_32 = coeffs.v_32;
        v_13 = coeffs.v_13;
        v_23 = coeffs.v_23;
        NEON_LOG_INFO("Homogenization complete");
    }
}

auto visualizer::Visualizer::SolveFEM(Real E, Real v) -> Real {
    const Vector3r force = Vector3r(0, y_axis_force_, 0);
    const MatrixXr pos_matrix = mesh_->RenderablePositions();
    NEON_LOG_INFO("Total nodes: ", pos_matrix.rows());
    // Apply uni-axial y-axis force
    // Bottom nodes are fixed
    const auto bottom_nodes = solvers::helpers::FindYAxisBottomNodes(pos_matrix);
    NEON_LOG_INFO("Bottom Nodes: ", bottom_nodes.size());

    // Top nodes have unit force
    const auto top_nodes = solvers::helpers::FindYAxisTopNodes(pos_matrix);

    // Since it's a cube, we can assume top nodes are all the same y.
    const Real fifty_percent_compression_threshold = pos_matrix.row(top_nodes.at(0)).y() / 2;

    std::vector<unsigned int> ignored_nodes;
    std::set_union(bottom_nodes.begin(), bottom_nodes.end(), top_nodes.begin(), top_nodes.end(),
                   std::back_inserter(ignored_nodes));
    const auto intermediate_nodes = solvers::helpers::SelectNodes(ignored_nodes, pos_matrix);

    const auto top_boundary_conditions = solvers::helpers::ApplyForceToBoundaryConditions(top_nodes, force);
    const auto intermediate_nodes_boundary_conditions =
            solvers::helpers::ApplyForceToBoundaryConditions(intermediate_nodes, force);

    auto all_boundary_conditions = top_boundary_conditions;
    all_boundary_conditions.insert(all_boundary_conditions.end(), intermediate_nodes_boundary_conditions.begin(),
                                   intermediate_nodes_boundary_conditions.end());

    NEON_LOG_INFO("Boundary Conditions (active dofs): ", all_boundary_conditions.size());
    fem_solver_ = std::make_unique<solvers::fem::LinearElastic>(all_boundary_conditions, E, v, mesh_);

    try {
        fem_solver_->SolveStatic();
    } catch (const std::string &ex) { NEON_LOG_ERROR(ex); }
    return fifty_percent_compression_threshold;
}

auto visualizer::Visualizer::GeometryMenuWindow() -> void {
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(geometry_menu_width_, -1.0f), ImVec2(geometry_menu_width_, -1.0f));
    bool _viewer_menu_visible = true;
    ImGui::Begin("Geometry Options", &_viewer_menu_visible,
                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.4f);
    GeometryMenu();
    ImGui::PopItemWidth();
    ImGui::End();
}

auto visualizer::Visualizer::GeometryMenu() -> void {
    if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        float p = ImGui::GetStyle().FramePadding.x;
        if (ImGui::Button("Load##Mesh", ImVec2((w - p) / 2.f, 0))) {
            const std::string f_name = igl::file_dialog_open();
            NEON_ASSERT_WARN(f_name.length() > 0, "File name empty.");
            const auto file_type = meshing::ReadFileExtension(f_name);

            if (mesh_ == nullptr) {
                if (tetrahedralize_) {
                    mesh_ = std::make_shared<meshing::Mesh>(f_name, tetgen_flags_, file_type);
                } else {
                    mesh_ = std::make_shared<meshing::Mesh>(f_name, file_type);
                }
            } else {
                if (tetrahedralize_) {
                    mesh_->ReloadMesh(f_name, tetgen_flags_, file_type);
                } else {
                    mesh_->ReloadMesh(f_name, file_type);
                }
            }
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

        ImGui::Checkbox("Tetrahedralize", &tetrahedralize_);
        ImGui::Checkbox("Isotropic", &isotropic_);

        if (ImGui::Button("Generate##Shape Generator", ImVec2((w - p) / 2.f, 0))) { GenerateShape(); }
    }
}

auto visualizer::Visualizer::GeneratorMenuWindow() -> void {
    const float x = geometry_menu_width_;
    ImGui::SetNextWindowPos(ImVec2(x, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(x, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(generator_menu_width_, -1.0f), ImVec2(generator_menu_width_, -1.0f));
    bool _viewer_menu_visible = true;
    ImGui::Begin("Generator Options", &_viewer_menu_visible,
                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.4f);
    GeneratorMenu();
    ImGui::PopItemWidth();
    ImGui::End();
}

auto visualizer::Visualizer::GeneratorMenu() -> void {
    if (ImGui::CollapsingHeader("Homogenization", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        float p = ImGui::GetStyle().FramePadding.x;

        ImGui::InputDouble("Young's Modulus", &youngs_modulus_);
        ImGui::InputDouble("Poisson's Ratio", &poissons_ratio_);

        ImGui::LabelText("E", "%.0e", youngs_modulus_);
        ImGui::LabelText("v", "%.0e", poissons_ratio_);

        ImGui::LabelText("E_x", "%.0e", E_x);
        ImGui::LabelText("E_y", "%.0e", E_y);
        ImGui::LabelText("E_z", "%.0e", E_z);
        ImGui::LabelText("G_x", "%.0e", G_x);
        ImGui::LabelText("G_y", "%.0e", G_y);
        ImGui::LabelText("G_z", "%.0e", G_z);
        ImGui::LabelText("v_21", "%.0e", v_21);
        ImGui::LabelText("v_31", "%.0e", v_31);
        ImGui::LabelText("v_12", "%.0e", v_12);
        ImGui::LabelText("v_32", "%.0e", v_32);
        ImGui::LabelText("v_13", "%.0e", v_13);
        ImGui::LabelText("v_23", "%.0e", v_23);

        if (ImGui::Button("Homogenize##Homogenization", ImVec2((w - p) / 2.f, 0))) { HomogenizeCurrentGeometry(); }
    }

    if (ImGui::CollapsingHeader("Dataset Generator", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        float p = ImGui::GetStyle().FramePadding.x;

        ImGui::InputInt("Samples", &n_samples_);

        if (ImGui::Button("Generate##Dataset Generator", ImVec2((w - p) / 2.f, 0))) {
            const std::string filename = "Generator.csv";
            utilities::filesystem::CsvFile<std::string> csv(filename,
                                                            std::vector<std::string>{"Voxel", "Coefficients"});
            for (int i = 0; i < n_samples_; ++i) {
                GenerateShape();
                HomogenizeCurrentGeometry();
                const VectorXr coeffs = rve_->Homogenized()->CoefficientVector();
                std::stringstream ss;
                ss << coeffs.transpose();
                const std::vector<std::string> entries({rve_->Homogenized()->Voxel().ToString(), ss.str()});
                csv << entries;
            }
        }
    }

    if (ImGui::CollapsingHeader("FEM Solver", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        float p = ImGui::GetStyle().FramePadding.x;

        ImGui::InputDouble("Y-Axis Force", &y_axis_force_);

        if (ImGui::Button("Solve##FEM Solver", ImVec2((w - p) / 2.f, 0))) {
            if (mesh_ == nullptr) {
                NEON_LOG_ERROR("No mesh found.");
            } else {
                const Real halfway = SolveFEM(youngs_modulus_, poissons_ratio_);
                NEON_LOG_INFO("Halfway computation: ", halfway);
                mesh_->Update(fem_solver_->U);
                Refresh();
            }
        }
    }
}
auto visualizer::Visualizer::SetupMenus() -> void {
    menu_.callback_draw_custom_window = [&]() { GeneratorMenuWindow(); };
    menu_.callback_draw_viewer_window = [&]() { GeometryMenuWindow(); };
    viewer_.plugins.push_back(&menu_);
}
