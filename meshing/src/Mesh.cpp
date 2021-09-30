// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <filesystem>
#include <igl/boundary_facets.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/for_each.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/readPLY.h>
#include <igl/signed_distance.h>
#include <igl/unique_simplices.h>
#include <igl/voxel_grid.h>
#include <meshing/Mesh.h>
#include <utilities/math/LinearAlgebra.h>
#include <utilities/runtime/NeonLog.h>

meshing::Mesh::Mesh(const std::string &file_path, MeshFileType file_type) { ReloadMesh(file_path, file_type); }

meshing::Mesh::Mesh(const std::string &file_path, const std::string &tetgen_flags, MeshFileType file_type) {
    MatrixXr V;
    MatrixXi F;
    ReadFile(file_path, file_type, V, F);

    ReloadMesh(V, F, tetgen_flags);
}

meshing::Mesh::Mesh(const MatrixXr &V, const MatrixXi &F) { ReloadMesh(V, F); }

meshing::Mesh::Mesh(const MatrixXr &V, const MatrixXi &F, const std::string &tetgen_flags) {
    ReloadMesh(V, F, tetgen_flags);
}

auto meshing::Mesh::Update(const MatrixXr &change) -> void { positions = rest_positions + change; }

auto meshing::Mesh::ReloadMesh(const MatrixXr &V, const MatrixXi &F) -> void {
    tetgen_succeeded = true;

    // Faces are tetrahedra
    if (F.cols() == 4) {
        igl::boundary_facets(F, faces);
        tetrahedra = F;
    } else {
        faces = F;
    }

    positions = V;
    rest_positions = V;
}

auto meshing::Mesh::ReloadMesh(const MatrixXr &V, const MatrixXi &F, const std::string &tetgen_flags) -> void {
    MatrixXr TV;
    MatrixXi TF;
    MatrixXi TT;

    // Sometimes with voids tetgen gets overlapping faces which causes all kinds of meltdowns, this aovids that outcome.
    MatrixXi FF;
    igl::unique_simplices(F, FF);
    const int res = igl::copyleft::tetgen::tetrahedralize(V, FF, tetgen_flags, TV, TT, TF);

    if (res != 0) {
        if ((V.rows() > 0 && V.cols() == 3) && (F.rows() > 0 && F.cols() == 3)) {
            NEON_LOG_ERROR("Tetgen failed to tetrahedralize uniform_mesh. Falling back to surface uniform_mesh.");
            ReloadMesh(V, F);
            return;
        }
        NEON_LOG_ERROR("New uniform_mesh was empty! Reverting overwrite and keeping old uniform_mesh. Check tetgen!");
        tetgen_succeeded = false;
        return;
    } else {
        tetgen_succeeded = true;
    }

    igl::boundary_facets(TT, faces);
    tetrahedra = TT;
    positions = TV;
    rest_positions = TV;
}

auto meshing::Mesh::ReloadMesh(const std::string &file_path, MeshFileType file_type) -> void {
    MatrixXr V;
    MatrixXi F;
    ReadFile(file_path, file_type, V, F);
    ReloadMesh(V, F);
}

auto meshing::Mesh::ReloadMesh(const std::string &file_path, const std::string &tetgen_flags, MeshFileType file_type)
        -> void {
    MatrixXr V;
    MatrixXi F;
    ReadFile(file_path, file_type, V, F);
    ReloadMesh(V, F, tetgen_flags);
}

auto meshing::Mesh::ReadFile(const std::string &file_path, MeshFileType file_type, MatrixXr &V, MatrixXi &F) -> void {
    switch (file_type) {
        case MeshFileType::kObj:
            igl::readOBJ(file_path, V, F);
            break;
        case MeshFileType::kPly:
            NEON_LOG_WARN("THIS IS PRONE TO LEAKING MEMORY. USE WITH CAUTION");
            igl::readPLY(file_path, V, F);
            break;
        case MeshFileType::kOff:
            igl::readOFF(file_path, V, F);
            break;
        default:
            throw std::runtime_error("Unsupported file type.");
    }
}

auto meshing::Mesh::ResetMesh() -> void { positions = rest_positions; }

auto meshing::Mesh::ToScalarField(const int dim) -> Tensor3r {
    MatrixXr GV;
    RowVector3i dimensions;
    constexpr int pad = 1;

    RowVector3r min_ext = positions.colwise().minCoeff();
    RowVector3r max_ext = positions.colwise().maxCoeff();
    Eigen::AlignedBox<Real, 3> box;
    box.extend(min_ext.transpose());
    box.extend(max_ext.transpose());

    igl::voxel_grid(box, dim, pad, GV, dimensions);

    VectorXi indices;
    VectorXr binary_field;
    MatrixXr C, N;

    NEON_LOG_INFO("Computing signed distances for mesh");
    igl::signed_distance(GV, positions, faces, igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL, binary_field, indices, C, N);
    for (int row = 0; row < binary_field.rows(); ++row) {
        const Real b = binary_field(row);
        binary_field.row(row) << (b > 0 ? 1 : 0);
    }

    return Tensor3r::Expand(binary_field, dimensions.x(), dimensions.y(), dimensions.z());
}
