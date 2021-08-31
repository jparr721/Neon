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

namespace visualizer {
    bool isotropic = false;
    bool tetrahedralize = true;

    int rve_dims = 5;
    int n_voids = 0;
    int void_dims = 0;

    Real youngs_modulus_ = 10000;
    Real poissons_ratio_ = 0.3;

    std::string tetgen_flags = "zpq";

    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    std::shared_ptr<meshing::Mesh> mesh;

    std::unique_ptr<solvers::fem::LinearElastic> solver;
    std::unique_ptr<solvers::integrators::CentralDifferenceMethod> integrator;
}// namespace visualizer

auto visualizer::Viewer() -> igl::opengl::glfw::Viewer & { return viewer; }
auto visualizer::Menu() -> igl::opengl::glfw::imgui::ImGuiMenu & { return menu; }
auto visualizer::Mesh() -> std::shared_ptr<meshing::Mesh> & { return mesh; }

auto visualizer::GenerateShape() -> void {
    const auto rve = std::make_unique<solvers::materials::Rve>(
            Vector3i(rve_dims, rve_dims, rve_dims), solvers::materials::MaterialFromEandv(1, "m_1", 1000, 0.3));
    MatrixXr V;
    MatrixXi F;
    meshing::ImplicitSurfaceGenerator<Real>::Inclusion inclusion{n_voids, void_dims, void_dims, void_dims};

    if (n_voids == 0) {
        rve->ComputeUniformMesh(V, F);
    } else {
        rve->ComputeCompositeMesh(inclusion, isotropic, V, F);
    }

    if (mesh == nullptr) {
        if (tetrahedralize) {
            mesh = std::make_shared<meshing::Mesh>(V, F, tetgen_flags);
        } else {
            mesh = std::make_shared<meshing::Mesh>(V, F);
        }
    } else {
        if (tetrahedralize) {
            mesh->ReloadMesh(V, F, tetgen_flags);
        } else {
            mesh->ReloadMesh(V, F);
        }
    }

    viewer.data().set_mesh(mesh->positions, mesh->faces);
}

auto visualizer::GeometryMenu() -> void {
    if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        if (ImGui::Button("Save##Mesh", ImVec2(w, 0))) { viewer.open_dialog_save_mesh(); }
    }

    // Viewing options
    if (ImGui::CollapsingHeader("Viewing Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Center object", ImVec2(-1, 0))) {
            viewer.core().align_camera_center(viewer.data().V, viewer.data().F);
        }
        if (ImGui::Button("Snap canonical view", ImVec2(-1, 0))) { viewer.snap_to_canonical_quaternion(); }

        // Zoom
        ImGui::PushItemWidth(80 * menu.menu_scaling());
        ImGui::DragFloat("Zoom", &(viewer.core().camera_zoom), 0.05f, 0.1f, 20.0f);
        ImGui::PopItemWidth();
    }

    // Helper for setting viewport specific mesh options
    auto make_checkbox = [&](const char *label, unsigned int &option) {
        return ImGui::Checkbox(
                label, [&]() { return viewer.core().is_set(option); },
                [&](bool value) { return viewer.core().set(option, value); });
    };

    // Draw options
    if (ImGui::CollapsingHeader("Draw Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Checkbox("Face-based", &(viewer.data().face_based))) {
            viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
        }
        if (ImGui::Checkbox("Invert normals", &(viewer.data().invert_normals))) {
            viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
        }
        ImGui::PopItemWidth();
    }

    // Overlays
    if (ImGui::CollapsingHeader("Overlays", ImGuiTreeNodeFlags_DefaultOpen)) {
        make_checkbox("Wireframe", viewer.data().show_lines);
        make_checkbox("Fill", viewer.data().show_faces);
        make_checkbox("Show vertex labels", viewer.data().show_vertex_labels);
        make_checkbox("Show faces labels", viewer.data().show_face_labels);
        make_checkbox("Show extra labels", viewer.data().show_custom_labels);
    }

    // Shape Generator
    if (ImGui::CollapsingHeader("Shape Generator", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        float p = ImGui::GetStyle().FramePadding.x;
        ImGui::InputInt("Rve Dim", &rve_dims);
        ImGui::InputInt("Void Dim", &void_dims);
        ImGui::InputInt("N Voids", &n_voids);

        ImGui::Checkbox("Tetrahedralize", &tetrahedralize);

        if (tetrahedralize) { ImGui::InputText("Flags", tetgen_flags); }

        ImGui::Checkbox("Isotropic", &isotropic);

        if (ImGui::Button("Generate##Shape Generator", ImVec2((w - p) / 2.f, 0))) { GenerateShape(); }
    }
}

auto visualizer::SimulationMenu() -> void {}
auto visualizer::SimulationMenuWindow() -> void {}

auto visualizer::SetupSolver() -> void {
    const Vector3r initial_force = Vector3r::Zero();
    const Vector3r force = Vector3r(0, -1 * 100, 0);
    // Apply uni-axial y-axis force
    // Bottom nodes are fixed
    const auto fixed_nodes_ = solvers::helpers::FindYAxisBottomNodes(mesh->positions);

    // Top nodes have unit force
    const auto force_applied_nodes_ = solvers::helpers::FindYAxisTopNodes(mesh->positions);

    std::vector<unsigned int> ignored_nodes;
    std::set_union(fixed_nodes_.begin(), fixed_nodes_.end(), force_applied_nodes_.begin(), force_applied_nodes_.end(),
                   std::back_inserter(ignored_nodes));

    const auto intermediate_nodes = solvers::helpers::SelectNodes(ignored_nodes, mesh->positions);

    const auto top_boundary_conditions = solvers::helpers::ApplyForceToBoundaryConditions(force_applied_nodes_, force);
    const auto intermediate_nodes_boundary_conditions =
            solvers::helpers::ApplyForceToBoundaryConditions(intermediate_nodes, initial_force);

    auto all_boundary_conditions = top_boundary_conditions;

    if (!intermediate_nodes_boundary_conditions.empty()) {
        all_boundary_conditions.insert(all_boundary_conditions.end(), intermediate_nodes_boundary_conditions.begin(),
                                       intermediate_nodes_boundary_conditions.end());
    }

    solver = std::make_unique<solvers::fem::LinearElastic>(all_boundary_conditions, youngs_modulus_, poissons_ratio_,
                                                           mesh, solvers::fem::LinearElastic::Type::kDynamic);
    integrator = std::make_unique<solvers::integrators::CentralDifferenceMethod>(0.01, 5, solver->K_e, solver->U_e,
                                                                                 solver->F_e);
}

auto visualizer::DrawCallback(igl::opengl::glfw::Viewer &) -> bool {
    if (viewer.core().is_animating) {
        integrator->Solve(solver->F_e, solver->U_e);
        solver->SolveWithIntegrator();
        viewer.data().set_mesh(mesh->positions, mesh->faces);
    }
    return false;
}