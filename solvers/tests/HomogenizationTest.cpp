// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#define BOOST_TEST_MODULE HomogenizationTests
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <solvers/materials/Homogenization.h>
#include <solvers/materials/Material.h>
#include <utilities/include/utilities/math/LinearAlgebra.h>

auto IsApprox(Real lhs, Real rhs, Real epsilon) -> bool { return std::fabs(lhs - rhs) < epsilon; }
auto ComputeSurfaceMesh() -> Tensor3r {
    Tensor3r t(10, 10, 10);
    t.SetConstant(1);
    return t;
}

BOOST_AUTO_TEST_CASE(TestHexahedron) {
    const auto material = solvers::materials::Material{1};
    const auto surface_mesh = ComputeSurfaceMesh();

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    BOOST_REQUIRE(homogenization.get() != nullptr);

    const auto hexahedron = homogenization->ComputeHexahedron(0.5, 0.5, 0.5);
    BOOST_REQUIRE_EQUAL(hexahedron.size(), 4);
}

BOOST_AUTO_TEST_CASE(TestComputeDegreesOfFreedom) {
    const auto material = solvers::materials::Material{1};
    const auto surface_mesh = ComputeSurfaceMesh();

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    BOOST_REQUIRE(homogenization.get() != nullptr);

    const MatrixX<int> edof = homogenization->ComputeElementDegreesOfFreedom(1000);

    BOOST_REQUIRE_EQUAL(edof.rows(), 1000);
    BOOST_REQUIRE_EQUAL(edof.cols(), 24);
    MatrixX<int> row_0_comp(1, 24);
    row_0_comp << 4, 5, 6, 37, 38, 39, 34, 35, 36, 1, 2, 3, 367, 368, 369, 400, 401, 402, 397, 398, 399, 364, 365, 366;
    BOOST_REQUIRE(edof.row(0).isApprox(row_0_comp));

    MatrixX<int> row_last_comp(1, 24);
    row_last_comp << 3595, 3596, 3597, 3628, 3629, 3630, 3625, 3626, 3627, 3592, 3593, 3594, 3958, 3959, 3960, 3991,
            3992, 3993, 3988, 3989, 3990, 3955, 3956, 3957;
    BOOST_REQUIRE(edof.row(999).isApprox(row_last_comp));
}

BOOST_AUTO_TEST_CASE(TestComputeUniqueNodes) {
    const auto material = solvers::materials::Material{1};
    const auto surface_mesh = ComputeSurfaceMesh();

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    BOOST_REQUIRE(homogenization.get() != nullptr);

    const Tensor3i unique_nodes = homogenization->ComputeUniqueNodes(1000);

    const MatrixX<int> first_layer = unique_nodes.At(0);
    const MatrixX<int> last_layer = unique_nodes.At(unique_nodes.Dimension(2) - 1);

    BOOST_REQUIRE(first_layer.isApprox(last_layer));

    const int rows = unique_nodes.Dimension(0);
    const int cols = unique_nodes.Dimension(1);
    const int layers = unique_nodes.Dimension(2);

    // Ensure the mirroring worked as intended.
    for (int l = 0; l < layers; ++l) {
        const VectorX<int> top_row = unique_nodes.Row(l, 0);
        const VectorX<int> bottom_row = unique_nodes.Row(l, rows - 1);

        BOOST_REQUIRE(top_row.isApprox(bottom_row));

        const VectorX<int> left_col = unique_nodes.Col(l, 0);
        const VectorX<int> right_col = unique_nodes.Col(l, cols - 1);

        BOOST_REQUIRE(left_col.isApprox(right_col));
    }
}

BOOST_AUTO_TEST_CASE(TestComputeUniqueDegreesOfFreedom) {
    const auto material = solvers::materials::Material{1};
    const auto surface_mesh = ComputeSurfaceMesh();

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    BOOST_REQUIRE(homogenization.get() != nullptr);

    constexpr unsigned int n_elements = 1000;
    const MatrixX<int> edof = homogenization->ComputeElementDegreesOfFreedom(n_elements);
    const Tensor3i unique_nodes = homogenization->ComputeUniqueNodes(n_elements);
    const MatrixX<int> unique_dof = homogenization->ComputeUniqueDegreesOfFreedom(edof, unique_nodes);

    auto row_0_comp = VectorX<int>(24);
    row_0_comp << 4, 5, 6, 34, 35, 36, 31, 32, 33, 1, 2, 3, 304, 305, 306, 334, 335, 336, 331, 332, 333, 301, 302, 303;

    BOOST_REQUIRE(row_0_comp.transpose().isApprox(unique_dof.row(0)));

    auto row_n_comp = VectorX<int>(24);
    row_n_comp << 2971, 2972, 2973, 2701, 2702, 2703, 2728, 2729, 2730, 2998, 2999, 3000, 271, 272, 273, 1, 2, 3, 28,
            29, 30, 298, 299, 300;

    BOOST_REQUIRE(row_n_comp.transpose().isApprox(unique_dof.row(unique_dof.rows() - 1)));
}

BOOST_AUTO_TEST_CASE(TestAssembleStiffnessMatrix) {
    const auto material = solvers::materials::MaterialFromLameCoefficients(1, "one", 10, 10);
    const auto surface_mesh = ComputeSurfaceMesh();

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    BOOST_REQUIRE(homogenization.get() != nullptr);
    const auto hexahedron = homogenization->ComputeHexahedron(0.5, 0.5, 0.5);

    constexpr unsigned int n_elements = 1000;
    const MatrixX<int> edof = homogenization->ComputeElementDegreesOfFreedom(n_elements);
    const Tensor3i unique_nodes = homogenization->ComputeUniqueNodes(n_elements);
    const MatrixX<int> unique_dof = homogenization->ComputeUniqueDegreesOfFreedom(edof, unique_nodes);
    const MatrixXr K = homogenization->AssembleStiffnessMatrix(3000, unique_dof, hexahedron.at(0), hexahedron.at(1));

    BOOST_REQUIRE(IsApprox(K(0, 0), 44.4445, 0.0001));
    BOOST_REQUIRE(K(0, 1) < 0.0001);
}

BOOST_AUTO_TEST_CASE(TestAssembleLoadMatrix) {
    const auto material = solvers::materials::MaterialFromLameCoefficients(1, "one", 10, 10);
    const auto surface_mesh = ComputeSurfaceMesh();

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    BOOST_REQUIRE(homogenization.get() != nullptr);
    const auto hexahedron = homogenization->ComputeHexahedron(0.5, 0.5, 0.5);

    constexpr unsigned int n_elements = 1000;
    const MatrixX<int> edof = homogenization->ComputeElementDegreesOfFreedom(n_elements);
    const Tensor3i unique_nodes = homogenization->ComputeUniqueNodes(n_elements);
    const MatrixX<int> unique_dof = homogenization->ComputeUniqueDegreesOfFreedom(edof, unique_nodes);
    const MatrixXr F = homogenization->AssembleLoadMatrix(1000, 3000, unique_dof, hexahedron.at(2), hexahedron.at(3));

    for (int row = 0; row < F.rows(); ++row) {
        for (int col = 0; col < F.cols(); ++col) {
            // Whole thing ends up being basically 0
            BOOST_REQUIRE(F(row, col) < 0.00001);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestAssembleLoadMatrixWithVoids) {
    const auto material = solvers::materials::MaterialFromLameCoefficients(1, "one", 10, 10);

    MatrixXr surface(10, 10);
    surface.row(0) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    surface.row(1) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(2) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(3) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(4) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(5) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(6) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(7) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(8) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(9) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

    Tensor3r surface_mesh = Tensor3r::Replicate(surface, 10);

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    BOOST_REQUIRE(homogenization.get() != nullptr);
    const auto hexahedron = homogenization->ComputeHexahedron(0.5, 0.5, 0.5);

    constexpr unsigned int n_elements = 1000;
    const MatrixX<int> edof = homogenization->ComputeElementDegreesOfFreedom(n_elements);
    const Tensor3i unique_nodes = homogenization->ComputeUniqueNodes(n_elements);
    const MatrixX<int> unique_dof = homogenization->ComputeUniqueDegreesOfFreedom(edof, unique_nodes);
    const MatrixXr F = homogenization->AssembleLoadMatrix(1000, 3000, unique_dof, hexahedron.at(2), hexahedron.at(3));

    // Whole thing ends up being basically 0 in the sum
    BOOST_REQUIRE(F.sum() < 0.00001);
}

BOOST_AUTO_TEST_CASE(TestComputeDisplacement) {
    const auto material = solvers::materials::MaterialFromLameCoefficients(1, "one", 10, 10);
    const auto surface_mesh = ComputeSurfaceMesh();

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    BOOST_REQUIRE(homogenization.get() != nullptr);
    const auto hexahedron = homogenization->ComputeHexahedron(0.5, 0.5, 0.5);

    constexpr unsigned int n_elements = 1000;
    const MatrixX<int> edof = homogenization->ComputeElementDegreesOfFreedom(n_elements);
    const Tensor3i unique_nodes = homogenization->ComputeUniqueNodes(n_elements);
    const MatrixX<int> unique_dof = homogenization->ComputeUniqueDegreesOfFreedom(edof, unique_nodes);
    const SparseMatrixXr F =
            homogenization->AssembleLoadMatrix(1000, 3000, unique_dof, hexahedron.at(2), hexahedron.at(3));
    const SparseMatrixXr K =
            homogenization->AssembleStiffnessMatrix(3000, unique_dof, hexahedron.at(0), hexahedron.at(1));

    const MatrixXr X = homogenization->ComputeDisplacement(3000, K, F, unique_dof);

    for (int row = 0; row < X.rows(); ++row) {
        for (int col = 0; col < X.cols(); ++col) {
            // Whole thing ends up being basically 0
            BOOST_REQUIRE(X(row, col) < 0.00001);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestComputeDisplacementWithVoidNodes) {
    const auto material = solvers::materials::MaterialFromLameCoefficients(1, "one", 10, 10);

    MatrixXr surface(10, 10);
    surface.row(0) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    surface.row(1) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(2) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(3) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(4) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(5) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(6) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(7) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(8) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(9) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

    Tensor3r surface_mesh = Tensor3r::Replicate(surface, 10);

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    BOOST_REQUIRE(homogenization.get() != nullptr);
    const auto hexahedron = homogenization->ComputeHexahedron(0.5, 0.5, 0.5);

    constexpr unsigned int n_elements = 1000;
    const MatrixX<int> edof = homogenization->ComputeElementDegreesOfFreedom(n_elements);
    const Tensor3i unique_nodes = homogenization->ComputeUniqueNodes(n_elements);
    const MatrixX<int> unique_dof = homogenization->ComputeUniqueDegreesOfFreedom(edof, unique_nodes);
    const SparseMatrixXr F =
            homogenization->AssembleLoadMatrix(1000, 3000, unique_dof, hexahedron.at(2), hexahedron.at(3));
    const SparseMatrixXr K =
            homogenization->AssembleStiffnessMatrix(3000, unique_dof, hexahedron.at(0), hexahedron.at(1));

    const MatrixXr X = homogenization->ComputeDisplacement(3000, K, F, unique_dof);

    // Just compare some of the values.
    const Real c_1 = X(4, 0);
    const Real c_2 = X(4, 1);
    const Real c_3 = X(4, 2);

    BOOST_REQUIRE(IsApprox(c_1, -0.325241, 0.0001));
    BOOST_REQUIRE(IsApprox(c_2, -0.0561329, 0.0001));
    BOOST_REQUIRE(IsApprox(c_3, -0.0953439, 0.0001));
}

BOOST_AUTO_TEST_CASE(TestComputeUnitStrainParameters) {
    const auto material = solvers::materials::MaterialFromLameCoefficients(1, "one", 10, 10);
    const auto surface_mesh = ComputeSurfaceMesh();

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    BOOST_REQUIRE(homogenization.get() != nullptr);
    const auto hexahedron = homogenization->ComputeHexahedron(0.5, 0.5, 0.5);
    const Tensor3r strain_param = homogenization->ComputeUnitStrainParameters(1000, hexahedron);

    const MatrixXr l0 = strain_param.Layer(0);
    VectorXr row = l0.row(0);
    Real sum = row.sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(3), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(6), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(15), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(18), 1, 0.0001));

    const MatrixXr l1 = strain_param.Layer(1);
    row = l1.row(0);
    sum = row.sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(7), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(10), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(19), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(22), 1, 0.0001));

    const MatrixXr l2 = strain_param.Layer(2);
    row = l2.row(0);
    sum = row.sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(14), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(17), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(20), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(23), 1, 0.0001));

    const MatrixXr l3 = strain_param.Layer(3);
    row = l3.row(0);
    sum = l3.row(0).sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(6), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(9), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(18), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(21), 1, 0.0001));

    const MatrixXr l4 = strain_param.Layer(4);
    row = l4.row(0);
    sum = l4.row(0).sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(13), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(16), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(19), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(22), 1, 0.0001));

    const MatrixXr l5 = strain_param.Layer(5);
    row = l5.row(0);
    sum = l5.row(0).sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(12), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(15), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(18), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(21), 1, 0.0001));
}

BOOST_AUTO_TEST_CASE(TestComputeUnitStrainParametersWithVoids) {
    const auto material = solvers::materials::MaterialFromLameCoefficients(1, "one", 10, 10);

    MatrixXr surface(10, 10);
    surface.row(0) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    surface.row(1) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(2) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(3) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(4) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(5) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(6) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(7) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(8) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(9) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

    Tensor3r surface_mesh = Tensor3r::Replicate(surface, 10);

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    BOOST_REQUIRE(homogenization.get() != nullptr);
    const auto hexahedron = homogenization->ComputeHexahedron(0.5, 0.5, 0.5);
    const Tensor3r strain_param = homogenization->ComputeUnitStrainParameters(1000, hexahedron);

    const MatrixXr l0 = strain_param.Layer(0);
    VectorXr row = l0.row(0);
    Real sum = row.sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(3), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(6), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(15), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(18), 1, 0.0001));

    const MatrixXr l1 = strain_param.Layer(1);
    row = l1.row(0);
    sum = row.sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(7), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(10), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(19), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(22), 1, 0.0001));

    const MatrixXr l2 = strain_param.Layer(2);
    row = l2.row(0);
    sum = row.sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(14), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(17), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(20), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(23), 1, 0.0001));

    const MatrixXr l3 = strain_param.Layer(3);
    row = l3.row(0);
    sum = l3.row(0).sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(6), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(9), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(18), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(21), 1, 0.0001));

    const MatrixXr l4 = strain_param.Layer(4);
    row = l4.row(0);
    sum = l4.row(0).sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(13), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(16), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(19), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(22), 1, 0.0001));

    const MatrixXr l5 = strain_param.Layer(5);
    row = l5.row(0);
    sum = l5.row(0).sum();
    BOOST_REQUIRE(std::fabs(sum - 4) < 0.0001);
    BOOST_REQUIRE(IsApprox(row(12), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(15), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(18), 1, 0.0001));
    BOOST_REQUIRE(IsApprox(row(21), 1, 0.0001));
}

BOOST_AUTO_TEST_CASE(TestSolverStep) {
    const auto material = solvers::materials::MaterialFromLameCoefficients(1, "one", 10, 10);
    const auto surface_mesh = ComputeSurfaceMesh();
    auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);

    homogenization->Solve();
    const MatrixXr constitutive_tensor = homogenization->Stiffness();

    Matrix6r comp;
    comp.row(0) << 30, 10, 10, 0, 0, 0;
    comp.row(1) << 10, 30, 10, 0, 0, 0;
    comp.row(2) << 10, 10, 30, 0, 0, 0;
    comp.row(3) << 0, 0, 0, 10, 0, 0;
    comp.row(4) << 0, 0, 0, 0, 10, 0;
    comp.row(5) << 0, 0, 0, 0, 0, 10;

    BOOST_REQUIRE(constitutive_tensor.isApprox(comp));
}

BOOST_AUTO_TEST_CASE(TestSolverStepWithVoids) {
    const auto material = solvers::materials::MaterialFromLameCoefficients(1, "one", 10, 10);

    MatrixXr surface(10, 10);
    surface.row(0) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    surface.row(1) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(2) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(3) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(4) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(5) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(6) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(7) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(8) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(9) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

    Tensor3r surface_mesh = Tensor3r::Replicate(surface, 10);

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    homogenization->Solve();
    const MatrixXr C = homogenization->Stiffness();

    Matrix6r comp;
    comp.row(0) << 5.66474, 0.48163, 1.53659, 0, 0, 0;
    comp.row(1) << 0.48163, 5.66474, 1.53659, 0, 0, 0;
    comp.row(2) << 1.53659, 1.53659, 9.7683, 0, 0, 0;
    comp.row(3) << 0, 0, 0, 0.16672, 0, 0;
    comp.row(4) << 0, 0, 0, 0, 2.15082, 0;
    comp.row(5) << 0, 0, 0, 0, 0, 2.15082;

    BOOST_REQUIRE(C.isApprox(comp, 0.01));
}

BOOST_AUTO_TEST_CASE(TestSolverOnLargerMatrix) {
    const auto material = solvers::materials::MaterialFromLameCoefficients(1, "one", 10, 10);

    MatrixXr surface(20, 20);
    surface.row(0) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    surface.row(1) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(2) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(3) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(4) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(5) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(6) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(7) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(8) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(9) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    surface.row(10) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    surface.row(11) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(12) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(13) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(14) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(15) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(16) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(17) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(18) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    surface.row(19) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    Tensor3r surface_mesh = Tensor3r::Replicate(surface, 20);

    const auto homogenization = std::make_shared<solvers::materials::Homogenization>(surface_mesh, material);
    homogenization->Solve();
}