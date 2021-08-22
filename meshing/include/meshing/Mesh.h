// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_MESH_H
#define NEON_MESH_H

#include <string>
#include <utilities/math/LinearAlgebra.h>

namespace meshing {
    enum class MeshFileType {
        kObj = 0,
        kPly,
        kOff,
    };

    class Mesh {
    public:
        VectorXr positions;
        VectorXr rest_positions;

        MatrixXi faces;

        Mesh(const std::string &file_path, MeshFileType file_type);
        Mesh(const std::string &file_path, const std::string &tetgen_flags, MeshFileType file_type);
        Mesh(const MatrixXr &V, const MatrixXi &F);

        auto Update(const VectorXr &displacements) -> void;

        auto ReloadMesh(const MatrixXr &V, const MatrixXi &F);
        auto ReloadMesh(const MatrixXr &V, const MatrixXi &F, const std::string &tetgen_flags);
    };
}// namespace meshing


#endif//NEON_MESH_H
