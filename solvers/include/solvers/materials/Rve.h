// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_RVE_H
#define NEON_RVE_H

#include <memory>
#include <meshing/ImplicitSurfaceGenerator.h>
#include <meshing/MarchingCubes.h>
#include <solvers/materials/Homogenization.h>
#include <solvers/materials/Material.h>
#include <string>
#include <utilities/math/LinearAlgebra.h>
#include <utility>

namespace solvers::materials {
    class Rve {
    public:
        Rve(const Vector3<int> &size, Material material_1);
        Rve(const Vector3<int> &size, Material material_1, Material material_2);

        auto Homogenize() -> void;
        auto ComputeRenderableMesh(MatrixXr &V, MatrixX<int> &F) -> void;
        auto ComputeSurfaceMesh() -> void;
        auto ComputeSurfaceMesh(const Vector3<int> &inclusion_size, int n_inclusions, bool is_isotropic) -> void;

        auto Height() const noexcept -> unsigned int { return height_; }
        auto Width() const noexcept -> unsigned int { return width_; }
        auto Depth() const noexcept -> unsigned int { return depth_; }
        auto ConsitutiveTensor() const -> Matrix6r { return C_; }
        auto SurfaceMesh() const -> Tensor3r { return surface_mesh_; }
        auto PrimaryMaterial() const -> Material { return material_1_; }
        auto SecondaryMaterial() const -> Material { return material_2_; }
        auto Homogenized() const noexcept -> const std::unique_ptr<Homogenization> & { return homogenization_; }

    private:
        static constexpr unsigned int kCellLength = 1;

        bool is_homogenized_ = false;
        bool contains_surface_mesh_ = false;

        unsigned int height_ = 0;
        unsigned int width_ = 0;
        unsigned int depth_ = 0;

        Material material_1_;
        Material material_2_;

        std::unique_ptr<meshing::ImplicitSurfaceGenerator<Real>> generator_;
        std::unique_ptr<meshing::MarchingCubes> marching_cubes_;
        std::unique_ptr<Homogenization> homogenization_;

        Matrix6r C_;
        Tensor3r surface_mesh_;
    };
}// namespace solvers::materials


#endif//NEON_RVE_H
