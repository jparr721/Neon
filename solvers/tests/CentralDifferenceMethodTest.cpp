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
#include <memory>
#include <solvers/integrators/CentralDifferenceMethod.h>

auto MakeStiffnessMatrix() -> Matrix2r {
    Matrix2r stiffness;
    stiffness << 6, -2, -2, 4;
    return stiffness;
}

auto MakeMassMatrix() -> SparseMatrixXr {
    using T = Eigen::Triplet<Real>;
    SparseMatrixXr mass_matrix;
    mass_matrix.resize(2, 2);
    const auto ul = T(0, 0, 2);
    const auto ur = T(0, 1, 0);
    const auto ll = T(1, 0, 0);
    const auto lr = T(1, 1, 1);
    auto vals = std::vector{ul, ur, ll, lr};
    mass_matrix.setFromTriplets(vals.begin(), vals.end());
    return mass_matrix;
}

BOOST_AUTO_TEST_CASE(TestConstructor) {
    const Vector2r initial_displacement = Vector2r(0, 0);
    const Vector2r initial_forces = Vector2r(0.f, 10.f);
    const SparseMatrixXr stiffness = MakeStiffnessMatrix().sparseView();
    const SparseMatrixXr mass_matrix = MakeMassMatrix();

    const auto integrator = std::make_unique<solvers::integrators::CentralDifferenceMethod>(
            .28, 1, stiffness, initial_displacement, initial_forces);

    BOOST_REQUIRE(integrator->Acceleration().isApprox(Vector2r(0, 10)));

    BOOST_REQUIRE_NE(integrator.get(), nullptr);
}

BOOST_AUTO_TEST_CASE(TestSolver) {
    VectorXr displacement = Vector2r(0, 0);
    Vector2r forces = Vector2r(0, 10);
    const SparseMatrixXr stiffness = MakeStiffnessMatrix().sparseView();
    const SparseMatrixXr mass_matrix = MakeMassMatrix();

    const auto integrator = std::make_unique<solvers::integrators::CentralDifferenceMethod>(.28, mass_matrix, stiffness,
                                                                                            displacement, forces);
    integrator->ComputeRayleighDamping(0, 0, 0);

    for (int i = 0; i < 12; ++i) {
        integrator->Solve(forces, displacement);
        if (i == 0) {
            VectorXr compare(2);
            compare << 0, 0.392;
            BOOST_REQUIRE(displacement.isApprox(compare, 0.001));
        }
    }

    VectorXr compare(2);
    compare << 1.0223, 2.60083;
    BOOST_REQUIRE(compare.isApprox(displacement, 0.001));
}