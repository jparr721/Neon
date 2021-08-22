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
#include <igl/readOBJ.h>
#include <utilities/math/LinearAlgebra.h>

int main() {
    MatrixXr V;
    MatrixX<int> F;

    igl::readOBJ("../Assets/armadillo.obj", V, F);
    igl::opengl::glfw::Viewer viewer;

    viewer.data().set_mesh(V, F);
    viewer.launch();

    return 0;
}
