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
    auto Viewer() -> igl::opengl::glfw::Viewer &;
    auto Menu() -> igl::opengl::glfw::imgui::ImGuiMenu &;

    // Operations
    auto GenerateShape() -> void;
    auto DrawCallback(igl::opengl::glfw::Viewer &viewer) -> bool;
    auto Refresh() -> void;

    // UI
    auto GeometryMenu() -> void;
    auto SimulationMenu() -> void;
    auto SimulationMenuWindow() -> void;

    // Simulations
    /// \brief Generates a dataset which solves the double integral of the signed volume of the unknown
    /// continuous function S(E, v) where S is parameterized by E and v.
    auto GenerateDisplacementDataset(const std::string &filename) -> void;

    /// \brief Generates a dataset of the homogenized version of a void-perforated uniform_mesh. By doing this
    /// we are able to optimize the uniform_mesh ratio in the system to approximate the functionality of softer
    /// materials as a result of strategically weakening harder materials.
    auto GenerateHomogenizationDataset(const std::string &filename) -> void;
}// namespace visualizer
#endif//NEON_VISUALIZER_H
