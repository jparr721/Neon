// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#include <future>
#include <meshing/DofOptimizer.h>
#include <solvers/materials/Material.h>
#include <thread>
#include <unordered_map>
#include <utilities/filesystem/CsvFile.h>
#include <utilities/math/Time.h>
#include <visualizer/Visualizer.h>
#include <visualizer/controllers/SolverController.h>

namespace visualizer {
    bool isotropic = false;
    bool tetrahedralize = true;

    int rve_dims = 5;
    int n_voids = 0;
    int void_dims = 0;
    int thickness = 1;

    constexpr Real E_static = 10000;
    constexpr Real v_static = 0.3;
    constexpr Real G_static = E_static / (2 * (1 + v_static));

    Real youngs_modulus = E_static;
    Real poissons_ratio = v_static;

    Real min_displacement = 1e3;

    solvers::materials::OrthotropicMaterial material =
            solvers::materials::OrthotropicMaterial(E_static, v_static, G_static);

    std::string tetgen_flags = "Yzpq";
    std::string displacement_dataset_name = "";

    // DOF nodes
    std::vector<unsigned int> fixed_nodes;
    std::vector<unsigned int> force_applied_nodes;

    MatrixXr displacements;
    MatrixXr stresses;

    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    std::shared_ptr<meshing::Mesh> uniform_mesh;
    std::shared_ptr<meshing::Mesh> perforated_mesh;
    std::unique_ptr<solvers::materials::Rve> rve;

    std::unique_ptr<solvers::fem::LinearElastic> solver;
    std::unique_ptr<solvers::integrators::CentralDifferenceMethod> integrator;

    std::unique_ptr<visualizer::controllers::SolverController> solver_controller =
            std::make_unique<visualizer::controllers::SolverController>(rve_dims, void_dims, thickness);

    const Vector3r force = Vector3r(0, -1 * 100, 0);
}// namespace visualizer

auto visualizer::RveDims() -> int & { return rve_dims; }
auto visualizer::Viewer() -> igl::opengl::glfw::Viewer & { return viewer; }
auto visualizer::Menu() -> igl::opengl::glfw::imgui::ImGuiMenu & { return menu; }
auto visualizer::UniformMesh() -> std::shared_ptr<meshing::Mesh> & { return solver_controller->UniformMesh(); }
auto visualizer::PerforatedMesh() -> std::shared_ptr<meshing::Mesh> & { return perforated_mesh; }
auto visualizer::Rve() -> std::unique_ptr<solvers::materials::Rve> & { return rve; }

auto visualizer::GenerateShape() -> void {
    solver_controller->ReloadMeshes(rve_dims, void_dims, thickness);
    //    rve = std::make_unique<solvers::materials::Rve>(
    //            Vector3i(rve_dims, rve_dims, rve_dims),
    //            solvers::materials::MaterialFromEandv(1, "m_1", youngs_modulus, poissons_ratio));
    //    const auto inclusion = meshing::ImplicitSurfaceGenerator<Real>::MakeInclusion(n_voids, void_dims);
    //
    //    MatrixXr V;
    //    MatrixXi F;
    //    if (n_voids == 0) {
    //        rve->ComputeUniformMesh(V, F);
    //    } else {
    //        rve->ComputeCompositeMesh(inclusion, thickness, isotropic, V, F);
    //    }
    //
    //    NEON_ASSERT_WARN(rve->GeneratorInfo() == "success", "RVE Generator failed at max iterations");
    //
    //    if (solver_controller->UniformMesh() == nullptr) {
    //        if (tetrahedralize) {
    //            solver_controller->UniformMesh() = std::make_shared<meshing::Mesh>(V, F, tetgen_flags);
    //        } else {
    //            solver_controller->UniformMesh() = std::make_shared<meshing::Mesh>(V, F);
    //        }
    //    } else {
    //        if (tetrahedralize) {
    //            solver_controller->UniformMesh()->ReloadMesh(V, F, tetgen_flags);
    //        } else {
    //            solver_controller->UniformMesh()->ReloadMesh(V, F);
    //        }
    //    }
}

auto visualizer::UpdateShapeEffectiveCoefficients() -> void {
    rve->SetMaterial(solvers::materials::MaterialFromEandv(1, "m_1", youngs_modulus, poissons_ratio));
}

auto visualizer::GeometryMenu() -> void {
    // UniformMesh
    if (ImGui::CollapsingHeader("UniformMesh", ImGuiTreeNodeFlags_DefaultOpen)) {
        float w = ImGui::GetContentRegionAvailWidth();
        if (ImGui::Button("Save##UniformMesh", ImVec2(w, 0))) { Viewer().open_dialog_save_mesh(); }
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
        ImGui::InputInt("Thickness", &thickness);

        ImGui::Checkbox("Tetrahedralize", &tetrahedralize);

        if (tetrahedralize) { ImGui::InputText("Flags", tetgen_flags); }

        ImGui::Checkbox("Isotropic", &isotropic);

        if (ImGui::Button("Generate##Shape Generator", ImVec2(w, 0))) {
            GenerateShape();
            Refresh();
        }
        if (ImGui::Button("Reset##Shape Generator", ImVec2(w, 0))) {
            UniformMesh()->ResetMesh();
            PerforatedMesh()->ResetMesh();
            Refresh();
            SetupSolver();
        }

        if (ImGui::Button("Reset Static##Shape Generator", ImVec2(w, 0))) {
            UniformMesh()->ResetMesh();
            Refresh();
            SetupStaticSolver();
        }
    }
}

auto visualizer::SimulationMenu() -> void {
    const float w = ImGui::GetContentRegionAvailWidth();
    if (ImGui::Button("Reload Solver", ImVec2(w, 0))) {
        if (!UniformMesh()->tetgen_succeeded) { return; }
        SetupSolver();
        //        UpdateShapeEffectiveCoefficients();
        solver_controller->UniformMesh()->ResetMesh();
    }

    if (ImGui::Button("Reload Static Solver", ImVec2(w, 0))) {
        if (!UniformMesh()->tetgen_succeeded) { return; }
        SetupStaticSolver();
        UpdateShapeEffectiveCoefficients();
        UniformMesh()->ResetMesh();
    }

    if (ImGui::CollapsingHeader("Homogenization", ImGuiTreeNodeFlags_DefaultOpen)) {
        const float w = ImGui::GetContentRegionAvailWidth();
        ImGui::LabelText("Source E", "%.0e", youngs_modulus);
        ImGui::LabelText("Source v", "%.0e", poissons_ratio);

        ImGui::LabelText("E_x", "%.0e", material.E_x);
        ImGui::LabelText("E_y", "%.0e", material.E_y);
        ImGui::LabelText("E_z", "%.0e", material.E_z);
        ImGui::LabelText("G_yz", "%.0e", material.G_yz);
        ImGui::LabelText("G_zx", "%.0e", material.G_zx);
        ImGui::LabelText("G_xy", "%.0e", material.G_xy);
        ImGui::LabelText("v_yx", "%.0e", material.v_yx);
        ImGui::LabelText("v_zx", "%.0e", material.v_zx);
        ImGui::LabelText("v_xy", "%.0e", material.v_xy);
        ImGui::LabelText("v_zy", "%.0e", material.v_zy);
        ImGui::LabelText("v_xz", "%.0e", material.v_xz);
        ImGui::LabelText("v_yz", "%.0e", material.v_yz);
        ImGui::LabelText("Min Displacement", "%.3f", min_displacement);

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

    if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) {
        const float w = ImGui::GetContentRegionAvailWidth();

        ImGui::InputText("Filename", displacement_dataset_name);
        if (ImGui::Button("Compute Static##Simulation", ImVec2(w, 0))) {
            solver->Solve(displacements, stresses);
            Real sum = 0;
            for (const auto &f : force_applied_nodes) { sum += solver_controller->UniformMesh()->positions.row(f).y(); }
            sum /= force_applied_nodes.size();
            min_displacement = std::fmin(sum, min_displacement);
            Refresh();
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
                                                           solver_controller->UniformMesh(),
                                                           solvers::fem::LinearElastic::Type::kDynamic);
    NEON_LOG_INFO("Solver loaded");
    integrator = std::make_unique<solvers::integrators::CentralDifferenceMethod>(0.01, 5, solver->K_e, solver->U_e,
                                                                                 solver->F_e);
    NEON_LOG_INFO("Integrator loaded");
}

auto visualizer::SetupStaticSolver() -> void {
    min_displacement = 1e3;
    const auto all_boundary_conditions = ComputeActiveDofs();

    solver = std::make_unique<solvers::fem::LinearElastic>(all_boundary_conditions, youngs_modulus, poissons_ratio,
                                                           solver_controller->UniformMesh(),
                                                           solvers::fem::LinearElastic::Type::kStatic);
    NEON_LOG_INFO("Solver loaded");
}

auto visualizer::DrawCallback(igl::opengl::glfw::Viewer &) -> bool {
    if (viewer.core().is_animating) {
        integrator->Solve(solver->F_e, solver->U_e);
        solver->Solve(displacements, stresses);

        solver_controller->UniformMesh()->Update(displacements);
        //        perforated_mesh->Update(displacements);

        Real sum = 0;
        for (const auto &f : force_applied_nodes) { sum += solver_controller->UniformMesh()->positions.row(f).y(); }
        sum /= force_applied_nodes.size();
        min_displacement = std::fmin(sum, min_displacement);
        Refresh();
    }
    return false;
}

auto visualizer::Refresh() -> void {
    if (!(Viewer().data().V.size() == solver_controller->UniformMesh()->positions.size() &&
          Viewer().data().F.size() == solver_controller->UniformMesh()->faces.size())) {
        Viewer().data().clear();
    }

    Viewer().data().set_mesh(solver_controller->UniformMesh()->positions, solver_controller->UniformMesh()->faces);
}

auto visualizer::Homogenize() -> void { rve->Homogenize(); }

auto visualizer::GenerateDisplacementDataset(const std::string &filename) -> void {
    //    const auto all_boundary_conditions = ComputeActiveDofs();
    //
    //    utilities::filesystem::CsvFile<std::string> csv(filename, {"E", "v", "Displacement"});
    //#pragma omp parallel for
    //    for (int E = 1000; E < 40000; E += 100) {
    //        for (Real v = 0.0; v < 0.5; v += 0.01) {
    //            // Copy the solver_controller->UniformMesh() object
    //            const auto mesh_clone = std::make_shared<meshing::Mesh>(*solver_controller->UniformMesh());
    //            const auto static_solver = std::make_unique<solvers::fem::LinearElastic>(
    //                    all_boundary_conditions, E, v, mesh_clone, solvers::fem::LinearElastic::Type::kStatic);
    //            static_solver->SolveStatic();
    //
    //            Real sum = 0;
    //            for (const auto &f : force_applied_nodes) { sum += mesh_clone->positions.row(f).y(); }
    //            sum /= force_applied_nodes.size();
    //
    //            csv << std::vector<std::string>{std::to_string(E), std::to_string(v), std::to_string(sum)};
    //        }
    //    }
}

auto visualizer::GenerateHomogenizationDataset(const std::string &filename) -> void {
    //    // Size is fixed so we can reliably iterate the same group of things
    //    const std::unordered_map<int, int> sizes = {{1, 5}, {2, 3}, {3, 2}, {4, 1}, {5, 1}};
    //
    //    utilities::filesystem::CsvFile<std::string> csv(filename,
    //                                                    std::vector<std::string>{"Effective Coefficients", "volume%"});
    //    for (const auto &[size, dims] : sizes) {
    //        const auto all_boundary_conditions = ComputeActiveDofs();
    //        const auto mesh_clone = std::make_shared<meshing::Mesh>(*solver_controller->UniformMesh());
    //        const auto static_solver =
    //                std::make_unique<solvers::fem::LinearElastic>(all_boundary_conditions, youngs_modulus, poissons_ratio,
    //                                                              mesh_clone, solvers::fem::LinearElastic::Type::kStatic);
    //        const auto rve = std::make_unique<solvers::materials::Rve>(
    //                Vector3i(rve_dims, rve_dims, rve_dims),
    //                solvers::materials::MaterialFromEandv(1, "m_1", youngs_modulus, poissons_ratio));
    //        const auto inclusion = meshing::ImplicitSurfaceGenerator<Real>::MakeInclusion(size, dims);
    //
    //        MatrixXr V;
    //        MatrixXi F;
    //        if (n_voids == 0) {
    //            rve->ComputeUniformMesh(V, F);
    //        } else {
    //            rve->ComputeCompositeMesh(inclusion, 0, isotropic, V, F);
    //        }
    //
    //        rve->Homogenize();
    //        std::stringstream ss;
    //        ss << rve->Homogenized()->CoefficientVector();
    //        const Real volume_pct =
    //                Tensor3r::SetConstant(1, rve->SurfaceMesh().Dimensions()).Sum() / rve->SurfaceMesh().Sum();
    //
    //        csv << std::vector<std::string>{ss.str(), std::to_string(volume_pct)};
    //    }
}

auto visualizer::ComputeActiveDofs() -> solvers::boundary_conditions::BoundaryConditions {
    std::vector<unsigned int> interior_nodes;
    meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, solver_controller->UniformMesh(), interior_nodes,
                                 force_applied_nodes, fixed_nodes);
    solvers::boundary_conditions::BoundaryConditions all_boundary_conditions;
    solvers::boundary_conditions::LoadBoundaryConditions(force, solver_controller->UniformMesh(), force_applied_nodes,
                                                         interior_nodes, all_boundary_conditions);

    return all_boundary_conditions;
}
