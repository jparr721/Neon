// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#include <datasets/Deformation.h>
#include <datasets/SolverMask.h>
#include <datasets/Static.h>
#include <future>
#include <thread>
#include <utilities/math/Time.h>
#include <visualizer/Visualizer.h>

namespace visualizer {
    bool isotropic = false;
    bool tetrahedralize = true;
    bool dataset_generating = false;

    int rve_dims = 50;
    Real amplitude = 0.160;
    Real thickness = 0.7;

    Real min_force = 1;
    Real max_force = 100;
    int dataset_shape = 10;

    const ImVec4 kErrorText(1, 0, 0, 1);
    const ImVec4 kOkayText(0, 1, 0, 1);

    std::string displacement_dataset_name = "";

    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    std::shared_ptr<visualizer::controllers::SolverController> solver_controller;

    const Vector3r force = Vector3r(0, -100, 0);
}// namespace visualizer

auto visualizer::Viewer() -> igl::opengl::glfw::Viewer & { return viewer; }
auto visualizer::Menu() -> igl::opengl::glfw::imgui::ImGuiMenu & { return menu; }
auto visualizer::Controller() -> std::shared_ptr<visualizer::controllers::SolverController> & {
    return solver_controller;
}
auto visualizer::RveDims() -> int & { return rve_dims; }
auto visualizer::Amplitude() -> Real & { return amplitude; }
auto visualizer::Thickness() -> Real & { return thickness; }

auto visualizer::GenerateShape() -> void {
    solver_controller->ReloadMeshes(rve_dims, amplitude, thickness);
    if (Viewer().data_list.empty() || Viewer().data_list.size() == 1) { viewer.append_mesh(true); }
    solver_controller->solvers_need_reload = true;
}

auto visualizer::GeometryMenu() -> void {
    // UniformMesh
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
        if (ImGui::Checkbox("Invert normals", &(Viewer().data().invert_normals))) {
            Viewer().data(controllers::SolverController::kUniformMeshID).invert_normals =
                    !Viewer().data(controllers::SolverController::kUniformMeshID).invert_normals;
            Viewer().data(controllers::SolverController::kUniformMeshID).dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
            Viewer().data(controllers::SolverController::kPerforatedMeshID).dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
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
        ImGui::InputDouble("Amplitude", &amplitude);
        ImGui::InputDouble("Thickness", &thickness);

        ImGui::Checkbox("Tetrahedralize", &tetrahedralize);

        if (tetrahedralize) { ImGui::InputText("Flags", solver_controller->TetgenFlags()); }

        ImGui::Checkbox("Isotropic", &isotropic);
        ImGui::Checkbox("Auto Compute Area", &solver_controller->SurfaceGeneratorAutoComputeArea());

        if (ImGui::Button("Generate##Shape Generator", ImVec2(w, 0))) {
            GenerateShape();
            Refresh();
        }

        if (ImGui::Button("Reset##Shape Generator", ImVec2(w, 0))) {
            solver_controller->ResetMeshPositions();
            Refresh();
            solver_controller->solvers_need_reload = true;
        }

        if (ImGui::Button("Reset Static##Shape Generator", ImVec2(w, 0))) {
            solver_controller->ResetMeshPositions();
            Refresh();
            solver_controller->solvers_need_reload = true;
        }
    }
}

auto visualizer::SimulationMenu() -> void {
    const float w = ImGui::GetContentRegionAvailWidth();
    if (ImGui::CollapsingHeader("Solvers", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputDouble("dt", &solver_controller->Dt());
        ImGui::InputDouble("Mass", &solver_controller->Mass());
        ImGui::InputDouble("Force", &solver_controller->Force());

        ImGui::InputDouble("E_x", &solver_controller->Material().E_x);
        ImGui::InputDouble("E_y", &solver_controller->Material().E_y);
        ImGui::InputDouble("E_z", &solver_controller->Material().E_z);
        ImGui::InputDouble("G_yz", &solver_controller->Material().G_yz);
        ImGui::InputDouble("G_zx", &solver_controller->Material().G_zx);
        ImGui::InputDouble("G_xy", &solver_controller->Material().G_xy);
        ImGui::InputDouble("v_yx", &solver_controller->Material().v_yx);
        ImGui::InputDouble("v_zx", &solver_controller->Material().v_zx);
        ImGui::InputDouble("v_xy", &solver_controller->Material().v_xy);
        ImGui::InputDouble("v_zy", &solver_controller->Material().v_zy);
        ImGui::InputDouble("v_xz", &solver_controller->Material().v_xz);
        ImGui::InputDouble("v_yz", &solver_controller->Material().v_yz);

        if (ImGui::Button("Reload Solver##Solvers", ImVec2(w, 0))) {
            solver_controller->ResetMeshPositions();
            if (!solver_controller->UniformMesh()->tetgen_succeeded) {
                NEON_LOG_WARN("Mesh was not tetrahedralized, cannot load solver!");
                return;
            }
            solver_controller->ReloadUniformSolver(solvers::fem::LinearElastic::Type::kDynamic);
            if (!solver_controller->PerforatedMesh()->tetgen_succeeded) {
                NEON_LOG_WARN("Mesh was not tetrahedralized, cannot load solver!");
                return;
            }
            solver_controller->ReloadPerforatedSolver(solvers::fem::LinearElastic::Type::kDynamic);
        }

        if (ImGui::Button("Reload Static Solver##Solvers", ImVec2(w, 0))) {
            solver_controller->ResetMeshPositions();
            if (!solver_controller->UniformMesh()->tetgen_succeeded) {
                NEON_LOG_WARN("Uniform Mesh was not tetrahedralized, cannot load solver!");
                return;
            }

            solver_controller->ReloadUniformSolver(solvers::fem::LinearElastic::Type::kStatic);

            if (!solver_controller->PerforatedMesh()->tetgen_succeeded) {
                NEON_LOG_WARN("Perforated Mesh was not tetrahedralized, cannot load solver!");
                return;
            }
            solver_controller->ReloadPerforatedSolver(solvers::fem::LinearElastic::Type::kStatic);
        }

        ImGui::TextColored(solver_controller->solvers_need_reload ? kErrorText : kOkayText,
                           solver_controller->solvers_need_reload ? "Solvers Need Reload" : "Solvers Ready");
    }

    if (ImGui::CollapsingHeader("Homogenization", ImGuiTreeNodeFlags_None)) {
        ImGui::InputDouble("Lambda", &solver_controller->Lambda());
        ImGui::InputDouble("Mu", &solver_controller->Mu());
        if (ImGui::Button("Homogenize##Homogenization", ImVec2(w, 0))) { solver_controller->HomogenizeVoidMesh(); }
    }

    if (ImGui::CollapsingHeader("Datasets", ImGuiTreeNodeFlags_None)) {
        ImGui::Text("Mask-Based Solver");
        ImGui::InputInt("Shape", &dataset_shape);
        ImGui::InputDouble("Min Force", &min_force);
        ImGui::InputDouble("Max Force", &max_force);
        if (ImGui::Button("Compute", ImVec2(w / 2, 0))) {
            NEON_LOG_INFO("Generating dataset in the background");
            auto task = std::thread(GenerateStaticSolverDataset);
            task.detach();
        }
        ImGui::TextColored(dataset_generating ? kOkayText : kErrorText, dataset_generating ? "Running" : "Stopped");
    }

    if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Compute Static##Simulation", ImVec2(w, 0))) {
            solver_controller->SolveUniform(controllers::SolverController::kUseStaticSolver);
            solver_controller->SolvePerforated(controllers::SolverController::kUseStaticSolver);
            Refresh();
        }
    }
}
auto visualizer::SimulationMenuWindow() -> void {
    const float x = 160.f * Menu().menu_scaling() + 20;
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

auto visualizer::DrawCallback(igl::opengl::glfw::Viewer &) -> bool {
    if (solver_controller->solvers_need_reload) { viewer.core().is_animating = false; }
    if (viewer.core().is_animating) {
        solver_controller->SolveUniform(controllers::SolverController::kUseDynamicSolver);
        solver_controller->SolvePerforated(controllers::SolverController::kUseDynamicSolver);
        Refresh();
    }
    return false;
}

auto visualizer::Refresh() -> void {
    // If the existing data in the mesh is the same size, then we don't need to worry about the "set_mesh" function since,
    // it'll efficiently handle the diff. However, if the sizes are different, we _must_ clear this info.
    if (!(Viewer().data(controllers::SolverController::kUniformMeshID).V.size() ==
                  solver_controller->UniformMesh()->positions.size() &&
          Viewer().data(controllers::SolverController::kUniformMeshID).F.size() ==
                  solver_controller->UniformMesh()->faces.size())) {
        Viewer().data(controllers::SolverController::kUniformMeshID).clear();
    }

    Viewer().data(controllers::SolverController::kUniformMeshID)
            .set_mesh(solver_controller->UniformMesh()->positions, solver_controller->UniformMesh()->faces);

    if (!(Viewer().data(controllers::SolverController::kPerforatedMeshID).V.size() ==
                  solver_controller->PerforatedMesh()->positions.size() &&
          Viewer().data(controllers::SolverController::kPerforatedMeshID).F.size() ==
                  solver_controller->PerforatedMesh()->faces.size())) {
        Viewer().data(controllers::SolverController::kPerforatedMeshID).clear();
    }

    Viewer().data(controllers::SolverController::kPerforatedMeshID)
            .set_mesh(solver_controller->PerforatedMesh()->positions, solver_controller->PerforatedMesh()->faces);
}

auto visualizer::GenerateSearchSpaceDataset() -> void {
    dataset_generating = true;
    auto deformation_generator = std::make_unique<datasets::Deformation>("deformation_dataset_generator.out");
    //    deformation_generator->GenerateSearchSpace()
}

auto visualizer::GenerateStaticSolverDataset() -> void {
    dataset_generating = true;
    datasets::MakeUniaxialStaticSolverDataset2D(min_force, max_force, dataset_shape);
    //    auto mask_dataset_generator = std::make_unique<datasets::DynamicSolverMask>(dataset_shape, n_entries);
    //    mask_dataset_generator->GenerateDataset(Vector3r(0, -100, 0), 5, 0.01, 30000, 0.3, 11538);
    dataset_generating = false;
}
