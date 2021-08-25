// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#include <Eigen/Core>
#include <filesystem>
#include <igl/opengl/glfw/Viewer.h>
#include <utilities/math/LinearAlgebra.h>
#include <solvers/materials/Material.h>
#include <solvers/materials/Rve.h>
#include <visualizer/Visualizer.h>

int main() {
// Igl's viewer requires vertex matrices to be doubles, fail if unset
#ifndef NEON_USE_DOUBLE
    throw std::runtime_exception("Please enable NEON_USE_DOUBLE to use igl viewer.");
#endif
    const auto rve = std::make_unique<solvers::materials::Rve>(
            Vector3i(51, 51, 51), solvers::materials::MaterialFromEandv(1, "mat_1", 10000, 0.3));
    MatrixXr V;
    MatrixXi F;
    rve->ComputeGridMesh(Vector3i(5, 5, 5), 20, true, V, F);
    const auto mesh = std::make_shared<meshing::Mesh>(V, F, "Yzpq");
    const auto visualizer = std::make_unique<visualizer::Visualizer>(mesh);
    visualizer->Launch();
}
