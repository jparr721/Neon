// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#define BOOST_TEST_MODULE LinearElasticTests
#define BOOST_DYN_TEST_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <meshing/Mesh.h>
#include <solvers/FEM/LinearElastic.h>
#include <utilities/math/LinearAlgebra.h>

auto MakeBasicMesh() -> std::shared_ptr<meshing::Mesh> {
    MatrixXr V(8, 3);
    V.row(0) << 0, 0, 0;
    V.row(1) << 0.025, 0, 0;
    V.row(2) << 0, 0.5, 0;
    V.row(3) << 0.025, 0.5, 0;
    V.row(4) << 0, 0, 0.25;
    V.row(5) << 0.025, 0, 0.25;
    V.row(6) << 0, 0.5, 0.25;
    V.row(7) << 0.025, 0.5, 0.25;

    MatrixXi T(5, 4);
    T.row(0) << 0, 1, 3, 5;
    T.row(1) << 0, 3, 2, 6;
    T.row(2) << 5, 4, 6, 0;
    T.row(3) << 5, 6, 7, 3;
    T.row(4) << 0, 5, 3, 6;

    return std::make_shared<meshing::Mesh>(V, T);
}

BOOST_AUTO_TEST_CASE(TestConstructor) {
    const auto mesh = MakeBasicMesh();
    const Real E = 210e6;
    const Real v = 0.3;
    const auto solver = std::make_unique<solvers::fem::LinearElastic>(
            solvers::boundary_conditions::BoundaryConditions{}, E, v, mesh);
    BOOST_REQUIRE(solver.get() != nullptr);
}

BOOST_AUTO_TEST_CASE(TestSolveStatic) {
    const auto mesh = MakeBasicMesh();
    const Real E = 210e6;
    const Real v = 0.3;
    const auto bc_1 = solvers::boundary_conditions::BoundaryCondition{
            2,
            Vector3r(0.f, 3.125f, 0.f),
    };

    const auto bc_2 = solvers::boundary_conditions::BoundaryCondition{
            3,
            Vector3r(0.f, 6.25f, 0.f),
    };

    const auto bc_3 = solvers::boundary_conditions::BoundaryCondition{
            6,
            Vector3r(0.f, 6.25f, 0.f),
    };

    const auto bc_4 = solvers::boundary_conditions::BoundaryCondition{
            7,
            Vector3r(0.f, 3.125f, 0.f),
    };
    solvers::boundary_conditions::BoundaryConditions bcs{bc_1, bc_2, bc_3, bc_4};
    const auto solver = std::make_unique<solvers::fem::LinearElastic>(bcs, E, v, mesh);
    BOOST_REQUIRE(solver.get() != nullptr);

    MatrixXr displacement;
    MatrixXr stress;
    solver->Solve(displacement, stress);
    MatrixXr stress_compare;
    stress_compare.resize(5, 6);
    stress_compare.row(0) << 1.47278, 3.43648, 1.47278, -0.0205161, 0.00896624, 0;
    stress_compare.row(1) << 0.00639241, 2.7694, 0.710224, -0.0128633, 0.0133638, -0.0703542;
    stress_compare.row(2) << 1.47278, 3.43648, 1.47278, 0.0205193, -.00896781, 0;
    stress_compare.row(3) << 0.00639215, 2.7694, 0.710224, 0.0128691, -.0133625, -0.0703543;
    stress_compare.row(4) << 0.00963581, 2.79405, 0.794476, -4.88281e-06, -1.38283e-07, 0.220376;

    BOOST_REQUIRE(stress.isApprox(stress_compare * 1e3, 0.001));
}