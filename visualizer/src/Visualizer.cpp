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

    Real youngs_modulus = 5000;
    Real poissons_ratio = 0.3;

    Real E_x = 0;
    Real E_y = 0;
    Real E_z = 0;
    Real G_x = 0;
    Real G_y = 0;
    Real G_z = 0;
    Real v_21 = 0;
    Real v_31 = 0;
    Real v_12 = 0;
    Real v_32 = 0;
    Real v_13 = 0;
    Real v_23 = 0;

    std::string tetgen_flags = "zpq";
    std::string displacement_dataset_name = "";

    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    std::shared_ptr<meshing::Mesh> mesh;

    std::unique_ptr<solvers::fem::LinearElastic> solver;
    std::unique_ptr<solvers::integrators::CentralDifferenceMethod> integrator;

    const Vector3r initial_force = Vector3r::Zero();
    const Vector3r force = Vector3r(0, -1 * 100, 0);
}// namespace visualizer

auto visualizer::RveDims() -> int & { return rve_dims; }
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

    NEON_LOG_INFO("Mesh change detected, reloading solver");
    SetupSolver();
    NEON_LOG_INFO("Solver ready to go!");
}

auto visualizer::GeometryMenu() -> void {
    // Mesh
    if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        if (ImGui::Button("Save##Mesh", ImVec2(w, 0))) { Viewer().open_dialog_save_mesh(); }
    }

    // Viewing options
    if (ImGui::CollapsingHeader("Viewing Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Center object", ImVec2(-1, 0))) {
            Viewer().core().align_camera_center(Viewer().data().V, Viewer().data().F);
        }
        if (ImGui::Button("Snap canonical view", ImVec2(-1, 0))) { Viewer().snap_to_canonical_quaternion(); }

        // Zoom
        ImGui::PushItemWidth(80 * menu.menu_scaling());
        ImGui::DragFloat("Zoom", &(Viewer().core().camera_zoom), 0.05f, 0.1f, 20.0f);
        ImGui::PopItemWidth();
    }

    // Helper for setting viewport specific mesh options
    auto make_checkbox = [&](const char *label, unsigned int &option) {
        return ImGui::Checkbox(
                label, [&]() { return Viewer().core().is_set(option); },
                [&](bool value) { return Viewer().core().set(option, value); });
    };

    // Draw options
    if (ImGui::CollapsingHeader("Draw Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Checkbox("Face-based", &(Viewer().data().face_based))) {
            Viewer().data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
        }
        if (ImGui::Checkbox("Invert normals", &(Viewer().data().invert_normals))) {
            Viewer().data().dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
        }
        ImGui::Checkbox("Animate", &(Viewer().core().is_animating));
    }


    // Overlays
    if (ImGui::CollapsingHeader("Overlays", ImGuiTreeNodeFlags_DefaultOpen)) {
        make_checkbox("Wireframe", Viewer().data().show_lines);
        make_checkbox("Fill", Viewer().data().show_faces);
        make_checkbox("Show vertex labels", Viewer().data().show_vertex_labels);
        make_checkbox("Show faces labels", Viewer().data().show_face_labels);
    }

    // Shape Generator
    if (ImGui::CollapsingHeader("Shape Generator", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        ImGui::InputInt("Rve Dim", &rve_dims);
        ImGui::InputInt("Void Dim", &void_dims);
        ImGui::InputInt("N Voids", &n_voids);

        ImGui::Checkbox("Tetrahedralize", &tetrahedralize);

        if (tetrahedralize) { ImGui::InputText("Flags", tetgen_flags); }

        ImGui::Checkbox("Isotropic", &isotropic);

        if (ImGui::Button("Generate##Shape Generator", ImVec2(w, 0))) {
            GenerateShape();
            Refresh();
        }
        if (ImGui::Button("Reset##Shape Generator", ImVec2(w, 0))) {
            Mesh()->ResetMesh();
            Refresh();
        }
    }
}

auto visualizer::SimulationMenu() -> void {
    if (ImGui::CollapsingHeader("Homogenization", ImGuiTreeNodeFlags_DefaultOpen)) {
        const float w = ImGui::GetContentRegionAvailWidth();

        ImGui::InputDouble("Young's Modulus", &youngs_modulus);
        ImGui::InputDouble("Poisson's Ratio", &poissons_ratio);

        ImGui::LabelText("E", "%.0e", youngs_modulus);
        ImGui::LabelText("v", "%.0e", poissons_ratio);

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

        if (ImGui::Button("Homogenize##Homogenization", ImVec2(w, 0))) { Homogenize(); }
    }

    if (ImGui::CollapsingHeader("Datasets", ImGuiTreeNodeFlags_DefaultOpen)) {
        const float w = ImGui::GetContentRegionAvailWidth();

        if (ImGui::CollapsingHeader("Displacement", ImGuiTreeNodeFlags_None)) {
            ImGui::InputText("Filename", displacement_dataset_name);
            if (ImGui::Button("Compute##Displacement", ImVec2(w, 0))) {
                if (!displacement_dataset_name.empty()) {
                    GenerateDisplacementDataset(displacement_dataset_name);
                } else {
                    NEON_LOG_ERROR("No filename for dataset!");
                }
            }
        }
    }
}
auto visualizer::SimulationMenuWindow() -> void {
    const float x = 160.f * Menu().menu_scaling();
    ImGui::SetNextWindowPos(ImVec2(x, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(x, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(x, -1.0f), ImVec2(x, -1.0f));
    bool _viewer_menu_visible = true;
    ImGui::Begin("Generator Options", &_viewer_menu_visible,
                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.4f);
    SimulationMenu();
    ImGui::PopItemWidth();
    ImGui::End();
}

auto visualizer::SetupSolver() -> void {
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

    solver = std::make_unique<solvers::fem::LinearElastic>(all_boundary_conditions, youngs_modulus, poissons_ratio,
                                                           mesh, solvers::fem::LinearElastic::Type::kDynamic);
    integrator = std::make_unique<solvers::integrators::CentralDifferenceMethod>(0.01, 5, solver->K_e, solver->U_e,
                                                                                 solver->F_e);
}

auto visualizer::DrawCallback(igl::opengl::glfw::Viewer &) -> bool {
    if (viewer.core().is_animating) {
        integrator->Solve(solver->F_e, solver->U_e);
        solver->SolveWithIntegrator();
        Refresh();
    }
    return false;
}

auto visualizer::Refresh() -> void {
    if (!(Viewer().data().V.size() == mesh->positions.size() && Viewer().data().F.size() == mesh->faces.size())) {
        Viewer().data().clear();
    }

    Viewer().data().set_mesh(mesh->positions, mesh->faces);
}

auto visualizer::Homogenize() -> void {}

auto visualizer::GenerateDisplacementDataset(const std::string &filename) -> void {
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

    utilities::filesystem::CsvFile<std::string> csv(filename, {"E", "v", "Displacement"});
#pragma omp parallel for
    for (int E = 1000; E < 40000; E += 100) {
        for (Real v = 0.0; v < 0.5; v += 0.01) {
            // Copy the mesh object
            const auto mesh_clone = std::make_shared<meshing::Mesh>(*mesh);
            const auto static_solver = std::make_unique<solvers::fem::LinearElastic>(
                    all_boundary_conditions, E, v, mesh_clone, solvers::fem::LinearElastic::Type::kStatic);
            static_solver->SolveStatic();

            Real sum = 0;
            for (const auto &f : force_applied_nodes_) { sum += mesh_clone->positions.row(f).y(); }
            sum /= force_applied_nodes_.size();

            csv << std::vector<std::string>{std::to_string(E), std::to_string(v), std::to_string(sum)};
        }
    }
}

auto visualizer::GenerateHomogenizationDataset(const std::string &filename) -> void {}
