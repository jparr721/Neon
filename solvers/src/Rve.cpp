// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <solvers/materials/Rve.h>

#include <utility>

solvers::materials::Rve::Rve(const Vector3<int> &size, Material material) : material_(std::move(material)) {
    height_ = size.x();
    width_ = size.y();
    depth_ = size.z();
}

auto solvers::materials::Rve::Homogenize() -> void {
    NEON_ASSERT_ERROR(contains_surface_mesh_, "No surface mesh found");
    homogenization_ = std::make_unique<Homogenization>(surface_mesh_, material_);
    homogenization_->Solve();
    C_ = homogenization_->Stiffness();
}


auto solvers::materials::Rve::ComputeCompositeMesh(const meshing::ImplicitSurfaceGenerator<Real>::Inclusion inclusion,
                                                   bool is_isotropic, MatrixXr &V, MatrixXi &F) -> void {
    const meshing::ImplicitSurfaceGenerator<Real>::Behavior behavior =
            is_isotropic ? meshing::ImplicitSurfaceGenerator<Real>::Behavior::kIsotropic
                         : meshing::ImplicitSurfaceGenerator<Real>::Behavior::kAnisotropic;

    generator_ =
            std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(height_, width_, depth_, behavior, inclusion);

    generator_->GenerateImplicitFunctionBasedMaterial(1, V, F);
    surface_mesh_ = generator_->Surface();

    contains_surface_mesh_ = true;
}

auto solvers::materials::Rve::ComputeUniformMesh() -> void {
    surface_mesh_ = Tensor3r(height_, width_, depth_);
    surface_mesh_.SetConstant(1);
    contains_surface_mesh_ = true;
}
auto solvers::materials::Rve::ComputeUnfiformMesh(MatrixXr &V, MatrixXi &F) -> void {
    generator_ = std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(height_, width_, depth_);
}
