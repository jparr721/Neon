// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_VISUALIZER_H
#define NEON_VISUALIZER_H

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui/imgui.h>
#include <meshing/Mesh.h>
#include <solvers/FEM/LinearElastic.h>
#include <solvers/integrators/CentralDifferenceMethod.h>
#include <utilities/math/LinearAlgebra.h>

namespace visualizer {
    class Visualizer {
    public:
        Visualizer();
        explicit Visualizer(std::shared_ptr<meshing::Mesh> mesh);

        [[nodiscard]] auto Viewer() const -> const igl::opengl::glfw::Viewer & { return viewer_; }
        [[nodiscard]] auto Menu() const -> const igl::opengl::glfw::imgui::ImGuiMenu & { return menu_; }

        auto Launch() -> void;
        auto Refresh() -> void;

        auto GenerateShape() -> void;
        auto HomogenizeCurrentGeometry() -> void;
        auto SolveStaticFEM() -> void;

    private:
        bool simulating_ = false;
        bool tetrahedralize_ = false;
        bool isotropic_ = true;

        int rve_dims_ = 20;
        int void_dims_ = 5;
        int n_voids_ = 10;
        int n_samples_ = 100;

        Real youngs_modulus_ = 10000;
        Real poissons_ratio_ = 0.3;

        Real y_axis_force_ = 100;
        const Vector3r initial_force = Vector3r::Zero();

        // Homogenization Coefficients =============================
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

        std::string tetgen_flags_ = "Yzpq";

        igl::opengl::glfw::Viewer viewer_;
        igl::opengl::glfw::imgui::ImGuiMenu menu_;

        std::shared_ptr<meshing::Mesh> mesh_;
        std::unique_ptr<solvers::materials::Rve> rve_;
        std::unique_ptr<solvers::fem::LinearElastic> fem_solver_;
        std::unique_ptr<solvers::integrators::CentralDifferenceMethod> fem_integrator_;

        const float geometry_menu_width_ = 160.f * menu_.menu_scaling();
        const float generator_menu_width_ = 200.f * menu_.menu_scaling();

        // This doesn't belong, but for the sake of time shall remain
        std::vector<unsigned int> fixed_nodes_;
        std::vector<unsigned int> dynamic_nodes_;
        std::vector<unsigned int> force_applied_nodes_;
        Real solver_force_distance_threshold_;

        // ==========================================================

        auto GeometryMenu() -> void;
        auto GeometryMenuWindow() -> void;

        auto GeneratorMenu() -> void;
        auto GeneratorMenuWindow() -> void;

        auto SetupMenus() -> void;

        auto SetupSimulator(solvers::fem::LinearElastic::Type sim_type) -> void;
        auto SetupIntegrator() -> void;

        auto SetActiveDofColors() -> void;
    };
}// namespace visualizer

#endif//NEON_VISUALIZER_H
