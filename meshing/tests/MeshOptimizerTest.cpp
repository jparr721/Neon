// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#define BOOST_TEST_MODULE MeshOptimizerTests
#define BOOST_DYN_TEST_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <igl/boundary_facets.h>
#include <igl/readPLY.h>
#include <iostream>
#include <meshing/MeshOptimizer.h>
#include <utilities/math/LinearAlgebra.h>

const std::string kAssetPath = "Assets/cube.ply";

BOOST_AUTO_TEST_CASE(TestComputeMeshNormals) {
    MatrixXr V;
    MatrixXi F;
    MatrixXr N;
    Vector3r D = Vector3r::Zero();
    igl::readPLY(kAssetPath, V, F);
    MatrixXi FF;
    igl::boundary_facets(F, FF);
    meshing::optimizer::ComputeNormals(V, FF, D, N);

    MatrixXr N_comp(N.rows(), N.cols());

    N_comp.row(0) << -1, 0, 0;
    N_comp.row(1) << 1, 0, 0;
    N_comp.row(2) << 0, 1, 0;
    N_comp.row(3) << 0, -1, 0;
    N_comp.row(4) << -1, 0, 0;
    N_comp.row(5) << 0, 0, -1;
    N_comp.row(6) << 0, 0, 1;
    N_comp.row(7) << 0, -1, 0;
    N_comp.row(8) << 0, 0, 1;
    N_comp.row(9) << 1, 0, 0;
    N_comp.row(10) << 0, 0, -1;
    N_comp.row(11) << 0, 0, 1;
    N_comp.row(12) << 0, 1, 0;
    N_comp.row(13) << 0, 0, 1;
    N_comp.row(14) << 0, -1, 0;
    N_comp.row(15) << 0, 1, 0;
    N_comp.row(16) << 0, 0, -1;
    N_comp.row(17) << 0, 1, 0;
    N_comp.row(18) << 0, 0, -1;
    N_comp.row(19) << 0, -1, 0;
    N_comp.row(20) << -1, 0, 0;
    N_comp.row(21) << 1, 0, 0;
    N_comp.row(22) << -1, 0, 0;
    N_comp.row(23) << 1, 0, 0;

    BOOST_REQUIRE(N_comp.isApprox(N));
}

BOOST_AUTO_TEST_CASE(TestComputeAdjacencies) {
    MatrixXr V;
    MatrixXi F;
    igl::readPLY(kAssetPath, V, F);
    SparseMatrixXi A;
    meshing::optimizer::ComputeAdjacencies(F, A);

    MatrixXi AA(A.rows(), A.cols());
    AA.row(0) << 0, 1, 1, 1, 1, 1, 0, 1;
    AA.row(1) << 1, 0, 1, 1, 1, 1, 1, 0;
    AA.row(2) << 1, 1, 0, 1, 0, 1, 1, 1;
    AA.row(3) << 1, 1, 1, 0, 1, 0, 1, 1;
    AA.row(4) << 1, 1, 0, 1, 0, 1, 1, 1;
    AA.row(5) << 1, 1, 1, 0, 1, 0, 1, 1;
    AA.row(6) << 0, 1, 1, 1, 1, 1, 0, 1;
    AA.row(7) << 1, 0, 1, 1, 1, 1, 1, 0;
    SparseMatrixXi A_comp = AA.sparseView();

    BOOST_REQUIRE(A_comp.isApprox(A));
}

BOOST_AUTO_TEST_CASE(TestComputeEdges) {
    MatrixXr V;
    MatrixXi F;
    igl::readPLY(kAssetPath, V, F);

    MatrixXi E;
    meshing::optimizer::ComputeEdges(V, F, E);

    MatrixXi E_comp(E.rows(), 2);
    E_comp.row(0) << 0, 1;
    E_comp.row(1) << 0, 2;
    E_comp.row(2) << 1, 2;
    E_comp.row(3) << 0, 3;
    E_comp.row(4) << 1, 3;
    E_comp.row(5) << 2, 3;
    E_comp.row(6) << 0, 4;
    E_comp.row(7) << 1, 4;
    E_comp.row(8) << 3, 4;
    E_comp.row(9) << 0, 5;
    E_comp.row(10) << 1, 5;
    E_comp.row(11) << 2, 5;
    E_comp.row(12) << 4, 5;
    E_comp.row(13) << 1, 6;
    E_comp.row(14) << 2, 6;
    E_comp.row(15) << 3, 6;
    E_comp.row(16) << 4, 6;
    E_comp.row(17) << 5, 6;
    E_comp.row(18) << 0, 7;
    E_comp.row(19) << 2, 7;
    E_comp.row(20) << 3, 7;
    E_comp.row(21) << 4, 7;
    E_comp.row(22) << 5, 7;
    E_comp.row(23) << 6, 7;

    BOOST_REQUIRE(E_comp.isApprox(E));
}

BOOST_AUTO_TEST_CASE(TestComputeEdgeLengths) {
    MatrixXr V(3, 3);
    V.row(0) << 1, -3, 0;
    V.row(1) << 0, 1, 0;
    V.row(2) << 1, 10, 0;

    MatrixXi F(1, 3);
    F.row(0) << 0, 1, 2;

    MatrixXr L;
    meshing::optimizer::ComputeEdgeLengths(V, F, L);

    const RowVector3r r = L.row(0);
    const Real a = r.x();
    const Real b = r.y();
    const Real c = r.z();

    using namespace utilities::math;
    BOOST_REQUIRE(IsApprox(a, 9.06, 0.01));
    BOOST_REQUIRE(IsApprox(b, 13, 0.01));
    BOOST_REQUIRE(IsApprox(c, 4.12, 0.01));
}

BOOST_AUTO_TEST_CASE(TestComputeTriangleSquaredArea) {
    MatrixXr V(3, 3);
    V.row(0) << 1, -3, 0;
    V.row(1) << 0, 1, 0;
    V.row(2) << 1, 10, 0;

    MatrixXi F(1, 3);
    F.row(0) << 0, 1, 2;

    MatrixXr L;
    meshing::optimizer::ComputeEdgeLengths(V, F, L);

    const RowVector3r r = L.row(0);
    const Real a = r.x();
    const Real b = r.y();
    const Real c = r.z();

    using namespace utilities::math;
    BOOST_REQUIRE(IsApprox(a, 9.06, 0.01));
    BOOST_REQUIRE(IsApprox(b, 13, 0.01));
    BOOST_REQUIRE(IsApprox(c, 4.12, 0.01));

    VectorXr A;
    meshing::optimizer::ComputeTriangleSquaredArea(L, A);

    BOOST_REQUIRE(IsApprox(A(0), 42.25, 0.01));
}