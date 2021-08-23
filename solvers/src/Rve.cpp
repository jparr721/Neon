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

solvers::materials::Rve::Rve(const Vector3<int> &size, Material material) : material_1_(std::move(material)) {
    height_ = size.x();
    width_ = size.y();
    depth_ = size.z();
}

solvers::materials::Rve::Rve(const Vector3<int> &size, Material material_1, Material material_2)
    : material_1_(std::move(material_1)), material_2_(std::move(material_2)) {
    height_ = size.x();
    width_ = size.y();
    depth_ = size.z();
}

auto solvers::materials::Rve::Homogenize() -> void {
    NEON_ASSERT_ERROR(contains_surface_mesh_, "No surface mesh found");
    if (material_2_.IsInit()) {
        homogenization_ = std::make_unique<Homogenization>(surface_mesh_, material_1_, material_2_);
    } else {
        homogenization_ = std::make_unique<Homogenization>(surface_mesh_, material_1_);
    }

    homogenization_->Solve();
    C_ = homogenization_->Stiffness();
    is_homogenized_ = true;
}

auto solvers::materials::Rve::ComputeRenderableMesh(MatrixXr &V, MatrixX<int> &F) -> void {
    NEON_ASSERT_ERROR(contains_surface_mesh_, "No surface mesh found");
    // Need to add the padding layers so that way marching cubes works properly
    surface_mesh_ = generator_->AddSquarePaddingLayers();
    marching_cubes_ =
            std::make_unique<meshing::MarchingCubes>(material_1_.number, kCellLength, surface_mesh_.Instance().data());
    marching_cubes_->GenerateGeometry(height_ + 1, width_ + 1, depth_ + 1, V, F);
}

auto solvers::materials::Rve::ComputeSurfaceMesh() -> void {
    generator_ = std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(height_, width_, depth_, material_1_.number);

    surface_mesh_ = generator_->Generate();

    contains_surface_mesh_ = true;
}

auto solvers::materials::Rve::ComputeSurfaceMesh(const Vector3<int> &inclusion_size, const int n_inclusions,
                                                 const bool is_isotropic) -> void {
    const meshing::ImplicitSurfaceGenerator<Real>::Inclusion inclusion{
            n_inclusions, inclusion_size.x(), inclusion_size.x(), inclusion_size.y(), inclusion_size.z(),
    };

    const meshing::ImplicitSurfaceGenerator<Real>::ImplicitSurfaceMicrostructure microstructure =
            meshing::ImplicitSurfaceGenerator<Real>::ImplicitSurfaceMicrostructure::kComposite;

    const meshing::ImplicitSurfaceGenerator<Real>::ImplicitSurfaceCharacteristics characteristics =
            is_isotropic ? meshing::ImplicitSurfaceGenerator<Real>::ImplicitSurfaceCharacteristics::kIsotropic
                         : meshing::ImplicitSurfaceGenerator<Real>::ImplicitSurfaceCharacteristics::kAnisotropic;

    const unsigned int material_2_number = material_2_.IsInit() ? material_2_.number : 0;

    generator_ = std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(
            height_, width_, depth_, characteristics, microstructure, inclusion, material_1_.number, material_2_number);

    surface_mesh_ = generator_->Generate();

    contains_surface_mesh_ = true;
}

auto solvers::materials::Rve::ComputeGridMesh(const Vector3<int> &inclusion_size, const int n_inclusions,
                                                 const bool is_isotropic, MatrixXr& V, MatrixXi& F) -> void {
    const meshing::ImplicitSurfaceGenerator<Real>::Inclusion inclusion{
            n_inclusions, inclusion_size.x(), inclusion_size.x(), inclusion_size.y(), inclusion_size.z(),
    };

    const meshing::ImplicitSurfaceGenerator<Real>::ImplicitSurfaceMicrostructure microstructure =
            meshing::ImplicitSurfaceGenerator<Real>::ImplicitSurfaceMicrostructure::kComposite;

    const meshing::ImplicitSurfaceGenerator<Real>::ImplicitSurfaceCharacteristics characteristics =
            is_isotropic ? meshing::ImplicitSurfaceGenerator<Real>::ImplicitSurfaceCharacteristics::kIsotropic
                         : meshing::ImplicitSurfaceGenerator<Real>::ImplicitSurfaceCharacteristics::kAnisotropic;

    const unsigned int material_2_number = material_2_.IsInit() ? material_2_.number : 0;

    generator_ = std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(
            height_, width_, depth_, characteristics, microstructure, inclusion, material_1_.number, material_2_number);

    generator_->GenerateImplicitFunctionBasedMaterial(V, F);
    surface_mesh_ = generator_->Surface();

    contains_surface_mesh_ = true;
}