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
#include <utilities/math/LinearAlgebra.h>

namespace visualizer {
    class Visualizer {
    public:
        Visualizer();
        explicit Visualizer(std::shared_ptr<meshing::Mesh> mesh);

        [[nodiscard]] auto Viewer() const -> const igl::opengl::glfw::Viewer & { return viewer_; }
        [[nodiscard]] auto Menu() const -> const igl::opengl::glfw::imgui::ImGuiMenu & { return menu_; }

        auto Launch() -> void;

        auto AddObjectToViewer() -> void;

        auto Refresh() -> void;
        auto UpdateVertexPositions(const VectorXr &displacements) -> void;

    private:
        bool tetrahedralize_ = false;

        int rve_dims_ = 10;
        int void_dims_ = 10;
        int n_voids_ = 5;

        Real youngs_modulus_ = 1000;
        Real poissons_ratio_ = 0;

        std::string tetgen_flags_ = "Yzpq";

        igl::opengl::glfw::Viewer viewer_;
        igl::opengl::glfw::imgui::ImGuiMenu menu_;

        std::shared_ptr<meshing::Mesh> mesh_;
        std::unique_ptr<solvers::materials::Rve> rve_;

        auto GeneratorMenu() -> void;
    };
}// namespace visualizer

#endif//NEON_VISUALIZER_H
