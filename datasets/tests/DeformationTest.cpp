// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#define BOOST_TEST_MODULE DeformationDatasetTests
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <datasets/Deformation.h>
#include <filesystem>
#include <memory>
#include <meshing/include/meshing/implicit_surfaces/ImplicitSurfaceGenerator.h>
#include <solvers/materials/Material.h>

auto ComputeActiveDofs(const std::shared_ptr<meshing::Mesh> &mesh) -> solvers::boundary_conditions::BoundaryConditions {
    const Vector3r force(0, -100, 0);
    std::vector<unsigned int> fixed_nodes;
    std::vector<unsigned int> force_applied_nodes;
    std::vector<unsigned int> interior_nodes;
    meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, mesh, interior_nodes, force_applied_nodes,
                                 fixed_nodes);
    solvers::boundary_conditions::BoundaryConditions all_boundary_conditions;
    solvers::boundary_conditions::LoadBoundaryConditions(force, mesh, force_applied_nodes, interior_nodes,
                                                         all_boundary_conditions);

    return all_boundary_conditions;
}

BOOST_AUTO_TEST_CASE(TestConstructor) {
    const auto df = std::make_unique<datasets::Deformation>("path");
    BOOST_REQUIRE(df.get() != nullptr);
    std::filesystem::remove_all("path");
}

BOOST_AUTO_TEST_CASE(TestGenerator) {
    const Vector3i size(5, 5, 5);
    const auto material = solvers::materials::MaterialFromEandv(1, "m", 15000, 0.3);

    MatrixXr V;
    MatrixXi F;
    const auto generator =
            std::make_unique<meshing::implicit_surfaces::ImplicitSurfaceGenerator<Real>>(size.x(), size.y(), size.z());
    generator->GenerateImplicitFunctionBasedMaterial(
            meshing::implicit_surfaces::ImplicitSurfaceGenerator<Real>::kNoThickness, 0, V, F);

    // generate the uniform_mesh with tetrahedralized components.
    const auto mesh = std::make_shared<meshing::Mesh>(V, F, "Yzpq");

    const auto boundary_conditions = ComputeActiveDofs(mesh);

    const auto df = std::make_unique<datasets::Deformation>("somepath");
    df->Generate(boundary_conditions, mesh, 1000, 40000, 0.0, 0.5, 1000, 0.01);

    // Delete the file when we're done
    std::filesystem::remove_all("somepath");
}