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
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/readPLY.h>
#include <meshing/Mesh.h>
#include <utilities/math/LinearAlgebra.h>

meshing::Mesh::Mesh(const std::string &file_path, meshing::MeshFileType file_type) { ReloadMesh(file_path, file_type); }

meshing::Mesh::Mesh(const std::string &file_path, const std::string &tetgen_flags, meshing::MeshFileType file_type) {
    MatrixXr V;
    MatrixXi F;
    ReadFile(file_path, file_type, V, F);

    ReloadMesh(V, F, tetgen_flags);
}

meshing::Mesh::Mesh(const MatrixXr &V, const MatrixXi &F) {
    faces = F;
    positions = utilities::math::MatrixToVector(V);
    rest_positions = utilities::math::MatrixToVector(V);
}

auto meshing::Mesh::Update(const VectorXr &displacements) -> void {}

auto meshing::Mesh::ReloadMesh(const MatrixXr &V, const MatrixXi &F) -> void { ReloadMesh(V, F, "zpq"); }

auto meshing::Mesh::ReloadMesh(const MatrixXr &V, const MatrixXi &F, const std::string &tetgen_flags) -> void {
    MatrixXr TV;
    MatrixXi TF;
    MatrixXi TT;
    igl::copyleft::tetgen::tetrahedralize(V, F, tetgen_flags, TV, TT, TF);
    igl::boundary_facets(TT, faces);
    positions = utilities::math::MatrixToVector(TV);
    rest_positions = utilities::math::MatrixToVector(TV);
}

auto meshing::Mesh::ReloadMesh(const std::string &file_path, MeshFileType file_type) -> void {
    MatrixXr V;
    MatrixXi F;
    ReadFile(file_path, file_type, V, F);
    ReloadMesh(V, F);
}

auto meshing::Mesh::RenderablePositions() -> MatrixXr {
    return utilities::math::VectorToMatrix(positions, positions.rows() / 3, 3);
}
auto meshing::Mesh::ReadFile(const std::string &file_path, MeshFileType file_type, MatrixXr &V, MatrixXi &F) -> void {
    switch (file_type) {
        case MeshFileType::kObj:
            igl::readOBJ(file_path, V, F);
            break;
        case MeshFileType::kPly:
            igl::readPLY(file_path, V, F);
            break;
        case MeshFileType::kOff:
            igl::readOFF(file_path, V, F);
            break;
        default:
            throw std::runtime_error("Unsupported file type.");
    }
}
auto meshing::ReadFileExtension(const std::string &filename) -> meshing::MeshFileType {
    const auto extension = std::filesystem::path(filename).extension().string();
    if (extension == ".obj") {
        return meshing::MeshFileType::kObj;
    } else if (extension == ".ply") {
        return meshing::MeshFileType::kPly;
    } else if (extension == ".off") {
        return meshing::MeshFileType::kOff;
    }

    return meshing::MeshFileType::kUnsupported;
}
