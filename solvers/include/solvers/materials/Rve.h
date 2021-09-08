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
        Rve(const Vector3<int> &size, Material material);

        auto Homogenize() -> void;
        auto ComputeUniformMesh() -> void;
        auto ComputeUniformMesh(MatrixXr &V, MatrixXi &F) -> void;
        auto ComputeCompositeMesh(meshing::ImplicitSurfaceGenerator<Real>::Inclusion inclusion, int thickness,
                                  bool is_isotropic, MatrixXr &V, MatrixXi &F) -> void;

        auto ConsitutiveTensor() const -> Matrix6r { return C_; }
        auto SurfaceMesh() const -> Tensor3r { return surface_mesh_; }
        auto PrimaryMaterial() const -> Material { return material_; }
        auto Homogenized() const noexcept -> const std::unique_ptr<Homogenization> & { return homogenization_; }
        auto GeneratorInfo() const noexcept -> std::string {
            switch (generator_->Info()) {
                case meshing::ImplicitSurfaceGenerator<Real>::GeneratorInfo::kFailure:
                    return "failed";
                default:
                    return "success";
            }
        }

        auto SetMaterial(const Material &m) -> void { material_ = m; }

    private:
        bool contains_surface_mesh_ = false;

        unsigned int height_ = 0;
        unsigned int width_ = 0;
        unsigned int depth_ = 0;

        Material material_;

        std::unique_ptr<meshing::ImplicitSurfaceGenerator<Real>> generator_;
        std::unique_ptr<Homogenization> homogenization_;

        Matrix6r C_;
        Tensor3r surface_mesh_;
    };
}// namespace solvers::materials


#endif//NEON_RVE_H
