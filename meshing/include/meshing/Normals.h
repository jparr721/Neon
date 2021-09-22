// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_NORMALS_H
#define NEON_NORMALS_H

#include <igl/parallel_for.h>
#include <igl/per_face_normals.h>
#include <utilities/math/LinearAlgebra.h>

namespace meshing {
    void InvertNegativeNormals(const MatrixXr &V, const MatrixXi &F, MatrixXr &N);
}// namespace meshing

#endif//NEON_NORMALS_H
