// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <meshing/Normals.h>

void meshing::InvertNegativeNormals(const MatrixXr &V, const MatrixXi &F, MatrixXr &N) {
    igl::per_face_normals(V, F, N);
    igl::parallel_for(N.rows(), [&](const int i) {
        if (N.row(i).sum() < 0) {
            // There's a more efficient way to do this, but I'll let parallelism make up for
            // bad code :)
            N.row(i) = (N.row(i).array() * -1).matrix();
        }
    });
}