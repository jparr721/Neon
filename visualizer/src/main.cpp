// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#include <Eigen/Core>
#include <algorithm>
#include <filesystem>
#include <igl/opengl/glfw/Viewer.h>
#include <utilities/math/LinearAlgebra.h>
#include <visualizer/Visualizer.h>


int main() {
// Igl's viewer requires vertex matrices to be doubles, fail if unset
#ifndef NEON_USE_DOUBLE
    throw std::runtime_exception("Please enable NEON_USE_DOUBLE to use igl viewer.");
#endif
    visualizer::GenerateShape();
    visualizer::Viewer().data().set_mesh(visualizer::Mesh()->positions, visualizer::Mesh()->faces);
    visualizer::SetupSolver();
    visualizer::Viewer().callback_pre_draw = &visualizer::DrawCallback;
    visualizer::Viewer().core().is_animating = true;
    visualizer::Viewer().launch();
}
