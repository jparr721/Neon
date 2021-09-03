// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#include <future>
#include <solvers/materials/Material.h>
#include <thread>
#include <utilities/filesystem/CsvFile.h>
#include <utilities/math/Time.h>
#include <visualizer/Visualizer.h>

namespace visualizer {
    bool isotropic = false;
    bool tetrahedralize = true;

    int rve_dims = 5;
    int n_voids = 0;
    int void_dims = 0;

    Real youngs_modulus = 5000;
    Real poissons_ratio = 0.3;

    Real min_displacement = 1e3;

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

    // DOF nodes
    std::vector<unsigned int> fixed_nodes;
    std::vector<unsigned int> force_applied_nodes;

    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    std::shared_ptr<meshing::Mesh> mesh;
    std::unique_ptr<solvers::materials::Rve> rve;

    std::unique_ptr<solvers::fem::LinearElastic> solver;
    std::unique_ptr<solvers::integrators::CentralDifferenceMethod> integrator;

    const Vector3r initial_force = Vector3r::Zero();
    const Vector3r force = Vector3r(0, -1 * 100, 0);
}// namespace visualizer

auto visualizer::RveDims() -> int & { return rve_dims; }
auto visualizer::Viewer() -> igl::opengl::glfw::Viewer & { return viewer; }
auto visualizer::Menu() -> igl::opengl::glfw::imgui::ImGuiMenu & { return menu; }
auto visualizer::Mesh() -> std::shared_ptr<meshing::Mesh> & { return mesh; }
auto visualizer::Rve() -> std::unique_ptr<solvers::materials::Rve> & { return rve; }

auto visualizer::GenerateShape() -> void {
    rve = std::make_unique<solvers::materials::Rve>(
            Vector3i(rve_dims, rve_dims, rve_dims),
            solvers::materials::MaterialFromEandv(1, "m_1", youngs_modulus, poissons_ratio));
    const auto inclusion = meshing::ImplicitSurfaceGenerator<Real>::MakeInclusion(n_voids, void_dims);

    MatrixXr V;
    MatrixXi F;
    if (n_voids == 0) {
        rve->ComputeUniformMesh(V, F);
    } else {
        rve->ComputeCompositeMesh(inclusion, isotropic, V, F);
    }

    NEON_ASSERT_WARN(rve->GeneratorInfo() == "success", "RVE Generator failed at max iterations");

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
}

auto visualizer::UpdateShapeEffectiveCoefficients() -> void {
    rve = std::make_unique<solvers::materials::Rve>(
            Vector3i(rve_dims, rve_dims, rve_dims),
            solvers::materials::MaterialFromEandv(1, "m_1", youngs_modulus, poissons_ratio));
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
            SetupSolver();
        }
    }
}

auto visualizer::SimulationMenu() -> void {
    const float w = ImGui::GetContentRegionAvailWidth();
    ImGui::InputDouble("Young's Modulus", &youngs_modulus);
    ImGui::InputDouble("Poisson's Ratio", &poissons_ratio);

    if (ImGui::Button("Reload Solver", ImVec2(w, 0))) {
        if (!Mesh()->tetgen_succeeded) { return; }
        SetupSolver();
        UpdateShapeEffectiveCoefficients();
        Mesh()->ResetMesh();
    }

    if (ImGui::CollapsingHeader("Homogenization", ImGuiTreeNodeFlags_DefaultOpen)) {
        const float w = ImGui::GetContentRegionAvailWidth();
        ImGui::LabelText("Source E", "%.0e", youngs_modulus);
        ImGui::LabelText("Source v", "%.0e", poissons_ratio);

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
        ImGui::LabelText("Min Displacement", "%.3f", min_displacement);

        if (ImGui::Button("Homogenize##Homogenization", ImVec2(w, 0))) {
            Homogenize();
            youngs_modulus = E_x;
            poissons_ratio = v_21;
        }
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
    min_displacement = 1e3;
    const auto all_boundary_conditions = ComputeActiveDofs();

    solver = std::make_unique<solvers::fem::LinearElastic>(all_boundary_conditions, youngs_modulus, poissons_ratio,
                                                           mesh, solvers::fem::LinearElastic::Type::kDynamic);
    NEON_LOG_INFO("Solver loaded");
    integrator = std::make_unique<solvers::integrators::CentralDifferenceMethod>(0.01, 5, solver->K_e, solver->U_e,
                                                                                 solver->F_e);
    NEON_LOG_INFO("Integrator loaded");
}

auto visualizer::DrawCallback(igl::opengl::glfw::Viewer &) -> bool {
    if (viewer.core().is_animating) {
        integrator->Solve(solver->F_e, solver->U_e);
        solver->SolveWithIntegrator();
        Real sum = 0;
        for (const auto &f : force_applied_nodes) { sum += mesh->positions.row(f).y(); }
        sum /= force_applied_nodes.size();
        min_displacement = std::fmin(sum, min_displacement);
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

auto visualizer::Homogenize() -> void {
    rve->Homogenize();
    E_x = rve->Homogenized()->Coefficients().E_11;
    E_y = rve->Homogenized()->Coefficients().E_22;
    E_z = rve->Homogenized()->Coefficients().E_33;
    G_x = rve->Homogenized()->Coefficients().G_23;
    G_y = rve->Homogenized()->Coefficients().G_31;
    G_z = rve->Homogenized()->Coefficients().G_12;
    v_21 = rve->Homogenized()->Coefficients().v_21;
    v_31 = rve->Homogenized()->Coefficients().v_31;
    v_12 = rve->Homogenized()->Coefficients().v_12;
    v_32 = rve->Homogenized()->Coefficients().v_32;
    v_13 = rve->Homogenized()->Coefficients().v_13;
    v_23 = rve->Homogenized()->Coefficients().v_23;
}

auto visualizer::GenerateDisplacementDataset(const std::string &filename) -> void {
    const auto all_boundary_conditions = ComputeActiveDofs();

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
            for (const auto &f : force_applied_nodes) { sum += mesh_clone->positions.row(f).y(); }
            sum /= force_applied_nodes.size();

            csv << std::vector<std::string>{std::to_string(E), std::to_string(v), std::to_string(sum)};
        }
    }
}

auto visualizer::GenerateHomogenizationDataset(const std::string &filename) -> void {
    const auto all_boundary_conditions = ComputeActiveDofs();
    const auto mesh_clone = std::make_shared<meshing::Mesh>(*mesh);
    const auto static_solver =
            std::make_unique<solvers::fem::LinearElastic>(all_boundary_conditions, youngs_modulus, poissons_ratio,
                                                          mesh_clone, solvers::fem::LinearElastic::Type::kStatic);
    const auto rve = std::make_unique<solvers::materials::Rve>(
            Vector3i(rve_dims, rve_dims, rve_dims),
            solvers::materials::MaterialFromEandv(1, "m_1", youngs_modulus, poissons_ratio));
    const auto inclusion = meshing::ImplicitSurfaceGenerator<Real>::MakeInclusion(n_voids, void_dims);

    MatrixXr V;
    MatrixXi F;
    if (n_voids == 0) {
        rve->ComputeUniformMesh(V, F);
    } else {
        rve->ComputeCompositeMesh(inclusion, isotropic, V, F);
    }

    // Iterate the solution space
    // Void dims are uniform, so the cube of it * n_voids must be half the total volume (to avoid degenerate cubes);
    const int void_area = std::powf(void_dims, 3);
    const int cuboid_area = std::powf(rve_dims, 3);
    NEON_ASSERT_ERROR((void_area * n_voids) / 2 <= cuboid_area, "Voids could cause degenerate cube!");

    // Figure out how many we can make per size.
}

auto visualizer::ComputeActiveDofs() -> solvers::helpers::BoundaryConditions {
    // Apply uni-axial y-axis force
    // Bottom nodes are fixed
    fixed_nodes = solvers::helpers::FindYAxisBottomNodes(mesh->positions);

    // Top nodes have unit force
    force_applied_nodes = solvers::helpers::FindYAxisTopNodes(mesh->positions);

    std::vector<unsigned int> ignored_nodes;
    std::set_union(fixed_nodes.begin(), fixed_nodes.end(), force_applied_nodes.begin(), force_applied_nodes.end(),
                   std::back_inserter(ignored_nodes));

    const auto intermediate_nodes = solvers::helpers::SelectNodes(ignored_nodes, mesh->positions);

    const auto top_boundary_conditions = solvers::helpers::ApplyForceToBoundaryConditions(force_applied_nodes, force);
    const auto intermediate_nodes_boundary_conditions =
            solvers::helpers::ApplyForceToBoundaryConditions(intermediate_nodes, initial_force);

    auto all_boundary_conditions = top_boundary_conditions;

    if (!intermediate_nodes_boundary_conditions.empty()) {
        all_boundary_conditions.insert(all_boundary_conditions.end(), intermediate_nodes_boundary_conditions.begin(),
                                       intermediate_nodes_boundary_conditions.end());
    }

    return all_boundary_conditions;
}
