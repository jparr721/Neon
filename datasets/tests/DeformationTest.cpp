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
#include <solvers/materials/Rve.h>

auto ComputeActiveDofs(const std::shared_ptr<meshing::Mesh> &mesh) -> solvers::boundary_conditions::BoundaryConditions {
    // Apply uni-axial y-axis force
    // Bottom nodes are fixed
    const auto fixed_nodes = solvers::boundary_conditions::FindYAxisBottomNodes(mesh->positions);

    // Top nodes have unit force
    const auto force_applied_nodes = solvers::boundary_conditions::FindYAxisTopNodes(mesh->positions);

    std::vector<unsigned int> ignored_nodes;
    std::set_union(fixed_nodes.begin(), fixed_nodes.end(), force_applied_nodes.begin(), force_applied_nodes.end(),
                   std::back_inserter(ignored_nodes));

    const auto intermediate_nodes = solvers::boundary_conditions::SelectNodes(ignored_nodes, mesh->positions);

    const auto top_boundary_conditions =
            solvers::boundary_conditions::ApplyForceToBoundaryConditions(force_applied_nodes, Vector3r(0, -100, 0));
    const auto intermediate_nodes_boundary_conditions =
            solvers::boundary_conditions::ApplyForceToBoundaryConditions(intermediate_nodes, Vector3r::Zero());

    auto all_boundary_conditions = top_boundary_conditions;

    if (!intermediate_nodes_boundary_conditions.empty()) {
        all_boundary_conditions.insert(all_boundary_conditions.end(), intermediate_nodes_boundary_conditions.begin(),
                                       intermediate_nodes_boundary_conditions.end());
    }

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
    const auto rve = std::make_unique<solvers::materials::Rve>(size, material);

    MatrixXr V;
    MatrixXi F;
    rve->ComputeUniformMesh(V, F);
    // generate the mesh with tetrahedralized components.
    const auto mesh = std::make_shared<meshing::Mesh>(V, F, "Yzpq");

    const auto boundary_conditions = ComputeActiveDofs(mesh);

    const auto df = std::make_unique<datasets::Deformation>("somepath");
    df->Generate(boundary_conditions, mesh, 1000, 40000, 0.0, 0.5, 1000, 0.01);

    // Delete the file when we're done
    std::filesystem::remove_all("somepath");
}