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
    const auto solver = std::make_unique<solvers::fem::LinearElastic>(solvers::helpers::BoundaryConditions{}, E, v,
                                                                      mesh, solvers::fem::LinearElastic::Type::kStatic);

    BOOST_REQUIRE(solver.get() != nullptr);
}