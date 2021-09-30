// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <igl/grid.h>
#include <igl/marching_cubes.h>
#include <igl/parallel_for.h>
#include <meshing/implicit_surfaces/PeriodicGyroid.h>

void meshing::implicit_surfaces::ComputeImplicitGyroidDualContouring(
        const Real amplitude, const Real thickness, const unsigned int resolution,
        const meshing::implicit_surfaces::GyroidImplicitFunction &fn, const dGyroidImplicitFunction &dfn, MatrixXr &V,
        MatrixXi &F, Tensor3r &scalar_field) {
    NEON_ASSERT_ERROR(resolution != 0, "Resolution cannot be zero!");

    // Computes the gradient of the implicit function and its derivative
    const auto fn_grad = [&](const RowVector3r &pos) { return dfn(fn, pos).normalized(); };
}

void meshing::implicit_surfaces::ComputeImplicitGyroidMarchingCubes(Real amplitude, Real thickness,
                                                                    unsigned int resolution,
                                                                    const GyroidImplicitFunction &fn, MatrixXr &V,
                                                                    MatrixXi &F, Tensor3r &scalar_field) {
    MatrixXr GV;
    igl::grid(RowVector3r(resolution, resolution, resolution), GV);
    VectorXr GF(GV.rows());
    igl::parallel_for(GV.rows(), [&](const int i) { GF(i) = fn(amplitude, GV.row(i)); });
    igl::parallel_for(GF.rows(), [&](const int i) { GF(i) = GF(i) < 0 ? 0 : 1; });
    scalar_field = Tensor3r::Expand(GF.transpose().eval(), resolution, resolution, resolution);
    NEON_LOG_INFO(scalar_field);
    igl::marching_cubes(GF, GV, resolution, resolution, resolution, thickness, V, F);
}
