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
#include <utilities/math/LinearAlgebra.h>

namespace visualizer {
    class Visualizer {
    public:
        Visualizer();

        auto Viewer() const -> const igl::opengl::glfw::Viewer & { return viewer_; }

        auto Menu() const -> const igl::opengl::glfw::imgui::ImGuiMenu & { return menu_; }

        auto Launch() -> void;

        auto SetMesh(const MatrixXr &V, const MatrixXi &F) -> void;

        auto AddObjectToViewer() -> void;

    private:
        igl::opengl::glfw::Viewer viewer_;
        igl::opengl::glfw::imgui::ImGuiMenu menu_;
    };
}// namespace visualizer

#endif//NEON_VISUALIZER_H
