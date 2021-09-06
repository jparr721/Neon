// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <Eigen/OrderingMethods>
#include <Eigen/SparseQR>
#include <igl/slice.h>
#include <solvers/FEM/LinearElastic.h>
#include <utilities/runtime/NeonLog.h>
#include <utility>

solvers::fem::LinearElastic::LinearElastic(boundary_conditions::BoundaryConditions boundary_conditions,
                                           Real youngs_modulus, Real poissons_ratio,
                                           std::shared_ptr<meshing::Mesh> mesh, Type type)
    : boundary_conditions(std::move(boundary_conditions)), youngs_modulus_(youngs_modulus),
      poissons_ratio_(poissons_ratio), mesh_(std::move(mesh)) {
    // Since this is a linear solver, we can formulate all of our starting assets right away.
    AssembleConstitutiveMatrix();
    AssembleElementStiffness();
    AssembleGlobalStiffness();
    AssembleBoundaryForces();
    if (type == Type::kDynamic) {
        // Element nodes for the dynamic case so we only use active dofs.
        U_e = VectorXr::Zero(this->boundary_conditions.size() * 3);
    }

    // Global displacement is always for all nodes.
    U = VectorXr::Zero(mesh_->positions.rows() * 3);
}

auto solvers::fem::LinearElastic::SolveWithIntegrator() -> MatrixXr {
    U.setZero();
    // Iterate the boundary conditions, assigning only where active nodes exist.
    int i = 0;
    for (const auto &[node, _] : boundary_conditions) {
        U.segment(node * 3, 3) << U_e(i), U_e(i + 1), U_e(i + 2);
        i += 3;
    }

    mesh_->Update((utilities::math::VectorToMatrix(U, 3, U.rows() / 3).transpose()).eval());

    return ComputeElementStress();
}
auto solvers::fem::LinearElastic::SolveStatic() -> MatrixXr {
    NEON_LOG_INFO("Firing up static solver");
    Eigen::SparseQR<SparseMatrixXr, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(K_e);
    NEON_ASSERT_ERROR(solver.info() == Eigen::Success, "Solver failed to compute factorization");
    U_e = solver.solve(F_e);
    NEON_LOG_INFO("Displacement computation complete");

    int i = 0;
    for (const auto &[node, _] : boundary_conditions) {
        U.segment(node * 3, 3) << U_e(i), U_e(i + 1), U_e(i + 2);
        i += 3;
    }

    F = K * U;
    NEON_LOG_INFO("Solver done, computing nodal stresses");
    mesh_->Update((utilities::math::VectorToMatrix(U, 3, U.rows() / 3).transpose()).eval());
    return ComputeElementStress();
}

auto solvers::fem::LinearElastic::AssembleGlobalStiffness() -> void {
    using triple = Eigen::Triplet<Real>;
    // Because it's nxn in matrix form, the vector form is 3n
    const unsigned int size = mesh_->positions.rows() * 3;
    K.resize(size, size);

    std::vector<triple> triplets;
    for (const auto &element_stiffness : K_e_storage) {
        const Matrix12r k = element_stiffness.stiffness;
        const auto i = element_stiffness.tetrahedral(0);
        const auto j = element_stiffness.tetrahedral(1);
        const auto m = element_stiffness.tetrahedral(2);
        const auto n = element_stiffness.tetrahedral(3);

        triplets.emplace_back(triple(3 * i, 3 * i, k(0, 0)));
        triplets.emplace_back(triple(3 * i, 3 * i + 1, k(0, 1)));
        triplets.emplace_back(triple(3 * i, 3 * i + 2, k(0, 2)));
        triplets.emplace_back(triple(3 * i, 3 * j, k(0, 3)));
        triplets.emplace_back(triple(3 * i, 3 * j + 1, k(0, 4)));
        triplets.emplace_back(triple(3 * i, 3 * j + 2, k(0, 5)));
        triplets.emplace_back(triple(3 * i, 3 * m, k(0, 6)));
        triplets.emplace_back(triple(3 * i, 3 * m + 1, k(0, 7)));
        triplets.emplace_back(triple(3 * i, 3 * m + 2, k(0, 8)));
        triplets.emplace_back(triple(3 * i, 3 * n, k(0, 9)));
        triplets.emplace_back(triple(3 * i, 3 * n + 1, k(0, 10)));
        triplets.emplace_back(triple(3 * i, 3 * n + 2, k(0, 11)));

        triplets.emplace_back(triple(3 * i + 1, 3 * i, k(1, 0)));
        triplets.emplace_back(triple(3 * i + 1, 3 * i + 1, k(1, 1)));
        triplets.emplace_back(triple(3 * i + 1, 3 * i + 2, k(1, 2)));
        triplets.emplace_back(triple(3 * i + 1, 3 * j, k(1, 3)));
        triplets.emplace_back(triple(3 * i + 1, 3 * j + 1, k(1, 4)));
        triplets.emplace_back(triple(3 * i + 1, 3 * j + 2, k(1, 5)));
        triplets.emplace_back(triple(3 * i + 1, 3 * m, k(1, 6)));
        triplets.emplace_back(triple(3 * i + 1, 3 * m + 1, k(1, 7)));
        triplets.emplace_back(triple(3 * i + 1, 3 * m + 2, k(1, 8)));
        triplets.emplace_back(triple(3 * i + 1, 3 * n, k(1, 9)));
        triplets.emplace_back(triple(3 * i + 1, 3 * n + 1, k(1, 10)));
        triplets.emplace_back(triple(3 * i + 1, 3 * n + 2, k(1, 11)));

        triplets.emplace_back(triple(3 * i + 2, 3 * i, k(2, 0)));
        triplets.emplace_back(triple(3 * i + 2, 3 * i + 1, k(2, 1)));
        triplets.emplace_back(triple(3 * i + 2, 3 * i + 2, k(2, 2)));
        triplets.emplace_back(triple(3 * i + 2, 3 * j, k(2, 3)));
        triplets.emplace_back(triple(3 * i + 2, 3 * j + 1, k(2, 4)));
        triplets.emplace_back(triple(3 * i + 2, 3 * j + 2, k(2, 5)));
        triplets.emplace_back(triple(3 * i + 2, 3 * m, k(2, 6)));
        triplets.emplace_back(triple(3 * i + 2, 3 * m + 1, k(2, 7)));
        triplets.emplace_back(triple(3 * i + 2, 3 * m + 2, k(2, 8)));
        triplets.emplace_back(triple(3 * i + 2, 3 * n, k(2, 9)));
        triplets.emplace_back(triple(3 * i + 2, 3 * n + 1, k(2, 10)));
        triplets.emplace_back(triple(3 * i + 2, 3 * n + 2, k(2, 11)));

        // j
        triplets.emplace_back(triple(3 * j, 3 * i, k(3, 0)));
        triplets.emplace_back(triple(3 * j, 3 * i + 1, k(3, 1)));
        triplets.emplace_back(triple(3 * j, 3 * i + 2, k(3, 2)));
        triplets.emplace_back(triple(3 * j, 3 * j, k(3, 3)));
        triplets.emplace_back(triple(3 * j, 3 * j + 1, k(3, 4)));
        triplets.emplace_back(triple(3 * j, 3 * j + 2, k(3, 5)));
        triplets.emplace_back(triple(3 * j, 3 * m, k(3, 6)));
        triplets.emplace_back(triple(3 * j, 3 * m + 1, k(3, 7)));
        triplets.emplace_back(triple(3 * j, 3 * m + 2, k(3, 8)));
        triplets.emplace_back(triple(3 * j, 3 * n, k(3, 9)));
        triplets.emplace_back(triple(3 * j, 3 * n + 1, k(3, 10)));
        triplets.emplace_back(triple(3 * j, 3 * n + 2, k(3, 11)));

        triplets.emplace_back(triple(3 * j + 1, 3 * i, k(4, 0)));
        triplets.emplace_back(triple(3 * j + 1, 3 * i + 1, k(4, 1)));
        triplets.emplace_back(triple(3 * j + 1, 3 * i + 2, k(4, 2)));
        triplets.emplace_back(triple(3 * j + 1, 3 * j, k(4, 3)));
        triplets.emplace_back(triple(3 * j + 1, 3 * j + 1, k(4, 4)));
        triplets.emplace_back(triple(3 * j + 1, 3 * j + 2, k(4, 5)));
        triplets.emplace_back(triple(3 * j + 1, 3 * m, k(4, 6)));
        triplets.emplace_back(triple(3 * j + 1, 3 * m + 1, k(4, 7)));
        triplets.emplace_back(triple(3 * j + 1, 3 * m + 2, k(4, 8)));
        triplets.emplace_back(triple(3 * j + 1, 3 * n, k(4, 9)));
        triplets.emplace_back(triple(3 * j + 1, 3 * n + 1, k(4, 10)));
        triplets.emplace_back(triple(3 * j + 1, 3 * n + 2, k(4, 11)));

        triplets.emplace_back(triple(3 * j + 2, 3 * i, k(5, 0)));
        triplets.emplace_back(triple(3 * j + 2, 3 * i + 1, k(5, 1)));
        triplets.emplace_back(triple(3 * j + 2, 3 * i + 2, k(5, 2)));
        triplets.emplace_back(triple(3 * j + 2, 3 * j, k(5, 3)));
        triplets.emplace_back(triple(3 * j + 2, 3 * j + 1, k(5, 4)));
        triplets.emplace_back(triple(3 * j + 2, 3 * j + 2, k(5, 5)));
        triplets.emplace_back(triple(3 * j + 2, 3 * m, k(5, 6)));
        triplets.emplace_back(triple(3 * j + 2, 3 * m + 1, k(5, 7)));
        triplets.emplace_back(triple(3 * j + 2, 3 * m + 2, k(5, 8)));
        triplets.emplace_back(triple(3 * j + 2, 3 * n, k(5, 9)));
        triplets.emplace_back(triple(3 * j + 2, 3 * n + 1, k(5, 10)));
        triplets.emplace_back(triple(3 * j + 2, 3 * n + 2, k(5, 11)));

        // m
        triplets.emplace_back(triple(3 * m, 3 * i, k(6, 0)));
        triplets.emplace_back(triple(3 * m, 3 * i + 1, k(6, 1)));
        triplets.emplace_back(triple(3 * m, 3 * i + 2, k(6, 2)));
        triplets.emplace_back(triple(3 * m, 3 * j, k(6, 3)));
        triplets.emplace_back(triple(3 * m, 3 * j + 1, k(6, 4)));
        triplets.emplace_back(triple(3 * m, 3 * j + 2, k(6, 5)));
        triplets.emplace_back(triple(3 * m, 3 * m, k(6, 6)));
        triplets.emplace_back(triple(3 * m, 3 * m + 1, k(6, 7)));
        triplets.emplace_back(triple(3 * m, 3 * m + 2, k(6, 8)));
        triplets.emplace_back(triple(3 * m, 3 * n, k(6, 9)));
        triplets.emplace_back(triple(3 * m, 3 * n + 1, k(6, 10)));
        triplets.emplace_back(triple(3 * m, 3 * n + 2, k(6, 11)));

        triplets.emplace_back(triple(3 * m + 1, 3 * i, k(7, 0)));
        triplets.emplace_back(triple(3 * m + 1, 3 * i + 1, k(7, 1)));
        triplets.emplace_back(triple(3 * m + 1, 3 * i + 2, k(7, 2)));
        triplets.emplace_back(triple(3 * m + 1, 3 * j, k(7, 3)));
        triplets.emplace_back(triple(3 * m + 1, 3 * j + 1, k(7, 4)));
        triplets.emplace_back(triple(3 * m + 1, 3 * j + 2, k(7, 5)));
        triplets.emplace_back(triple(3 * m + 1, 3 * m, k(7, 6)));
        triplets.emplace_back(triple(3 * m + 1, 3 * m + 1, k(7, 7)));
        triplets.emplace_back(triple(3 * m + 1, 3 * m + 2, k(7, 8)));
        triplets.emplace_back(triple(3 * m + 1, 3 * n, k(7, 9)));
        triplets.emplace_back(triple(3 * m + 1, 3 * n + 1, k(7, 10)));
        triplets.emplace_back(triple(3 * m + 1, 3 * n + 2, k(7, 11)));

        triplets.emplace_back(triple(3 * m + 2, 3 * i, k(8, 0)));
        triplets.emplace_back(triple(3 * m + 2, 3 * i + 1, k(8, 1)));
        triplets.emplace_back(triple(3 * m + 2, 3 * i + 2, k(8, 2)));
        triplets.emplace_back(triple(3 * m + 2, 3 * j, k(8, 3)));
        triplets.emplace_back(triple(3 * m + 2, 3 * j + 1, k(8, 4)));
        triplets.emplace_back(triple(3 * m + 2, 3 * j + 2, k(8, 5)));
        triplets.emplace_back(triple(3 * m + 2, 3 * m, k(8, 6)));
        triplets.emplace_back(triple(3 * m + 2, 3 * m + 1, k(8, 7)));
        triplets.emplace_back(triple(3 * m + 2, 3 * m + 2, k(8, 8)));
        triplets.emplace_back(triple(3 * m + 2, 3 * n, k(8, 9)));
        triplets.emplace_back(triple(3 * m + 2, 3 * n + 1, k(8, 10)));
        triplets.emplace_back(triple(3 * m + 2, 3 * n + 2, k(8, 11)));

        // n
        triplets.emplace_back(triple(3 * n, 3 * i, k(9, 0)));
        triplets.emplace_back(triple(3 * n, 3 * i + 1, k(9, 1)));
        triplets.emplace_back(triple(3 * n, 3 * i + 2, k(9, 2)));
        triplets.emplace_back(triple(3 * n, 3 * j, k(9, 3)));
        triplets.emplace_back(triple(3 * n, 3 * j + 1, k(9, 4)));
        triplets.emplace_back(triple(3 * n, 3 * j + 2, k(9, 5)));
        triplets.emplace_back(triple(3 * n, 3 * m, k(9, 6)));
        triplets.emplace_back(triple(3 * n, 3 * m + 1, k(9, 7)));
        triplets.emplace_back(triple(3 * n, 3 * m + 2, k(9, 8)));
        triplets.emplace_back(triple(3 * n, 3 * n, k(9, 9)));
        triplets.emplace_back(triple(3 * n, 3 * n + 1, k(9, 10)));
        triplets.emplace_back(triple(3 * n, 3 * n + 2, k(9, 11)));

        triplets.emplace_back(triple(3 * n + 1, 3 * i, k(10, 0)));
        triplets.emplace_back(triple(3 * n + 1, 3 * i + 1, k(10, 1)));
        triplets.emplace_back(triple(3 * n + 1, 3 * i + 2, k(10, 2)));
        triplets.emplace_back(triple(3 * n + 1, 3 * j, k(10, 3)));
        triplets.emplace_back(triple(3 * n + 1, 3 * j + 1, k(10, 4)));
        triplets.emplace_back(triple(3 * n + 1, 3 * j + 2, k(10, 5)));
        triplets.emplace_back(triple(3 * n + 1, 3 * m, k(10, 6)));
        triplets.emplace_back(triple(3 * n + 1, 3 * m + 1, k(10, 7)));
        triplets.emplace_back(triple(3 * n + 1, 3 * m + 2, k(10, 8)));
        triplets.emplace_back(triple(3 * n + 1, 3 * n, k(10, 9)));
        triplets.emplace_back(triple(3 * n + 1, 3 * n + 1, k(10, 10)));
        triplets.emplace_back(triple(3 * n + 1, 3 * n + 2, k(10, 11)));

        triplets.emplace_back(triple(3 * n + 2, 3 * i, k(11, 0)));
        triplets.emplace_back(triple(3 * n + 2, 3 * i + 1, k(11, 1)));
        triplets.emplace_back(triple(3 * n + 2, 3 * i + 2, k(11, 2)));
        triplets.emplace_back(triple(3 * n + 2, 3 * j, k(11, 3)));
        triplets.emplace_back(triple(3 * n + 2, 3 * j + 1, k(11, 4)));
        triplets.emplace_back(triple(3 * n + 2, 3 * j + 2, k(11, 5)));
        triplets.emplace_back(triple(3 * n + 2, 3 * m, k(11, 6)));
        triplets.emplace_back(triple(3 * n + 2, 3 * m + 1, k(11, 7)));
        triplets.emplace_back(triple(3 * n + 2, 3 * m + 2, k(11, 8)));
        triplets.emplace_back(triple(3 * n + 2, 3 * n, k(11, 9)));
        triplets.emplace_back(triple(3 * n + 2, 3 * n + 1, k(11, 10)));
        triplets.emplace_back(triple(3 * n + 2, 3 * n + 2, k(11, 11)));
    }

    K.setFromTriplets(triplets.begin(), triplets.end());
}

auto solvers::fem::LinearElastic::AssembleElementStiffness() -> void {
    for (int row = 0; row < mesh_->tetrahedra.rows(); ++row) {
        const Vector4i tetrahedral = mesh_->tetrahedra.row(row);

        // Get vertices corresponding to the tetrahedral node labels.
        const Vector3r shape_one = mesh_->positions.row(tetrahedral(0));
        const Vector3r shape_two = mesh_->positions.row(tetrahedral(1));
        const Vector3r shape_three = mesh_->positions.row(tetrahedral(2));
        const Vector3r shape_four = mesh_->positions.row(tetrahedral(3));

        // Prepare to compute the nodal stresses by transforming via the shape functions
        // and then computing the stress.
        const MatrixXr B = AssembleStrainRelationshipMatrix(shape_one, shape_two, shape_three, shape_four);
        const Real V = ComputeTetrahedralElementVolume(shape_one, shape_two, shape_three, shape_four);
        const Matrix12r stiffness = V * B.transpose() * constitutive_matrix_ * B;


        K_e_storage.emplace_back(ElementStiffness{stiffness, tetrahedral});
    }
}

auto solvers::fem::LinearElastic::AssembleBoundaryForces() -> void {
    NEON_ASSERT_WARN(!boundary_conditions.empty(),
                     "No boundary conditions found. This simulation will not run properly.");
    F_e = VectorXr::Zero(boundary_conditions.size() * 3);
    VectorXr boundary_force_indices(boundary_conditions.size() * 3);

    int segment = 0;
    for (const auto &[node, force] : boundary_conditions) {
        const auto _node = node * 3;
        F_e.segment(segment, 3) << force;
        boundary_force_indices.segment(segment, 3) << _node, _node + 1, _node + 2;
        segment += 3;
    }

    igl::slice(K, boundary_force_indices, boundary_force_indices, K_e);
}

auto solvers::fem::LinearElastic::AssemblePlaneStresses(const MatrixXr &sigmas) -> MatrixXr {
    MatrixXr plane_stresses;
    plane_stresses.resize(sigmas.rows(), 3);

    for (int row = 0; row < sigmas.rows(); ++row) {
        const VectorXr sigma = sigmas.row(row);
        const Real s1 = sigma.sum();
        const Real s2 = (sigma(0) * sigma(1) + sigma(0) * sigma(2) + sigma(1) * sigma(2)) -
                        (sigma(3) * sigma(3) - sigma(4) * sigma(4) - sigma(5) * sigma(5));

        Matrix3r ms3;
        ms3.row(0) << sigma(0), sigma(3), sigma(5);
        ms3.row(1) << sigma(3), sigma(1), sigma(4);
        ms3.row(2) << sigma(5), sigma(4), sigma(2);

        const Real s3 = ms3.determinant();

        const Vector3r plane_stress(s1, s2, s3);
        plane_stresses.row(row) = plane_stress;
    }

    return plane_stresses;
}
auto solvers::fem::LinearElastic::ComputeElementStress() -> MatrixXr {
    MatrixXr element_stresses;
    element_stresses.resize(mesh_->tetrahedra.rows(), 6);

    // Convert the positions vector to a matrix for easier indexing.
    // Note: for large geometry this could cause performance issues.
    const MatrixXr dsp = utilities::math::VectorToMatrix(U, 3, U.rows() / 3).transpose();
    for (int row = 0; row < mesh_->tetrahedra.rows(); ++row) {
        const Vector4i tetrahedral = mesh_->tetrahedra.row(row);

        // Get vertices corresponding to the tetrahedral node labels.
        const Vector3r shape_one = mesh_->positions.row(tetrahedral(0));
        const Vector3r shape_two = mesh_->positions.row(tetrahedral(1));
        const Vector3r shape_three = mesh_->positions.row(tetrahedral(2));
        const Vector3r shape_four = mesh_->positions.row(tetrahedral(3));

        // Get the corresponding node displacement values by tetrahedral index.
        const Vector3r displacement_one = dsp.row(tetrahedral(0));
        const Vector3r displacement_two = dsp.row(tetrahedral(1));
        const Vector3r displacement_three = dsp.row(tetrahedral(2));
        const Vector3r displacement_four = dsp.row(tetrahedral(3));

        // Prepare to compute the nodal stresses by transforming via the shape functions
        // and then computing the stress.
        const MatrixXr B = AssembleStrainRelationshipMatrix(shape_one, shape_two, shape_three, shape_four);
        Vector12r u;
        u << displacement_one, displacement_two, displacement_three, displacement_four;
        element_stresses.row(row) = constitutive_matrix_ * B * u;
    }

    return element_stresses;
}
auto solvers::fem::LinearElastic::ComputeShapeFunctionFromPoints(const Vector6r &points) -> Real {
    const Real p0 = points(0);
    const Real p1 = points(1);
    const Real p2 = points(2);
    const Real p3 = points(3);
    const Real p4 = points(4);
    const Real p5 = points(5);

    Matrix3r parameter;
    parameter.row(0) << 1, p0, p1;
    parameter.row(1) << 1, p2, p3;
    parameter.row(2) << 1, p4, p5;
    return parameter.determinant();
}

auto solvers::fem::LinearElastic::AssembleConstitutiveMatrix() -> void {
    constitutive_matrix_.row(0) << 1 - poissons_ratio_, poissons_ratio_, poissons_ratio_, 0, 0, 0;
    constitutive_matrix_.row(1) << poissons_ratio_, 1 - poissons_ratio_, poissons_ratio_, 0, 0, 0;
    constitutive_matrix_.row(2) << poissons_ratio_, poissons_ratio_, 1 - poissons_ratio_, 0, 0, 0;
    constitutive_matrix_.row(3) << 0, 0, 0, (1 - 2 * poissons_ratio_) / 2, 0, 0;
    constitutive_matrix_.row(4) << 0, 0, 0, 0, (1 - 2 * poissons_ratio_) / 2, 0;
    constitutive_matrix_.row(5) << 0, 0, 0, 0, 0, (1 - 2 * poissons_ratio_) / 2;
    constitutive_matrix_ *= youngs_modulus_ / ((1 + poissons_ratio_) * (1 - 2 * poissons_ratio_));
}

auto solvers::fem::LinearElastic::AssembleStrainRelationshipMatrix(const Vector3r &shape_one, const Vector3r &shape_two,
                                                                   const Vector3r &shape_three,
                                                                   const Vector3r &shape_four) -> MatrixXr {
    using Matrix63r = Eigen::Matrix<Real, 6, 3>;
    MatrixXr strain_relationship;
    const Real V = ComputeTetrahedralElementVolume(shape_one, shape_two, shape_three, shape_four);
    const auto create_beta_submatrix = [](Real beta, Real gamma, Real delta) -> Matrix63r {
        Matrix63r B;
        B.row(0) << beta, 0, 0;
        B.row(1) << 0, gamma, 0;
        B.row(2) << 0, 0, delta;
        B.row(3) << gamma, beta, 0;
        B.row(4) << 0, delta, gamma;
        B.row(5) << delta, 0, beta;
        return B;
    };

    const Real x1 = shape_one.x();
    const Real y1 = shape_one.y();
    const Real z1 = shape_one.z();

    const Real x2 = shape_two.x();
    const Real y2 = shape_two.y();
    const Real z2 = shape_two.z();

    const Real x3 = shape_three.x();
    const Real y3 = shape_three.y();
    const Real z3 = shape_three.z();

    const Real x4 = shape_four.x();
    const Real y4 = shape_four.y();
    const Real z4 = shape_four.z();

    const Real beta_1 = -1 * ComputeShapeFunctionFromPoints((Vector6r() << y2, z2, y3, z3, y4, z4).finished());
    const Real beta_2 = ComputeShapeFunctionFromPoints((Vector6r() << y1, z1, y3, z3, y4, z4).finished());
    const Real beta_3 = -1 * ComputeShapeFunctionFromPoints((Vector6r() << y1, z1, y2, z2, y4, z4).finished());
    const Real beta_4 = ComputeShapeFunctionFromPoints((Vector6r() << y1, z1, y2, z2, y3, z3).finished());

    const Real gamma_1 = ComputeShapeFunctionFromPoints((Vector6r() << x2, z2, x3, z3, x4, z4).finished());
    const Real gamma_2 = -1 * ComputeShapeFunctionFromPoints((Vector6r() << x1, z1, x3, z3, x4, z4).finished());
    const Real gamma_3 = ComputeShapeFunctionFromPoints((Vector6r() << x1, z1, x2, z2, x4, z4).finished());
    const Real gamma_4 = -1 * ComputeShapeFunctionFromPoints((Vector6r() << x1, z1, x2, z2, x3, z3).finished());

    const Real delta_1 = -1 * ComputeShapeFunctionFromPoints((Vector6r() << x2, y2, x3, y3, x4, y4).finished());
    const Real delta_2 = ComputeShapeFunctionFromPoints((Vector6r() << x1, y1, x3, y3, x4, y4).finished());
    const Real delta_3 = -1 * ComputeShapeFunctionFromPoints((Vector6r() << x1, y1, x2, y2, x4, y4).finished());
    const Real delta_4 = ComputeShapeFunctionFromPoints((Vector6r() << x1, y1, x2, y2, x3, y3).finished());

    const Matrix63r B1 = create_beta_submatrix(beta_1, gamma_1, delta_1);
    const Matrix63r B2 = create_beta_submatrix(beta_2, gamma_2, delta_2);
    const Matrix63r B3 = create_beta_submatrix(beta_3, gamma_3, delta_3);
    const Matrix63r B4 = create_beta_submatrix(beta_4, gamma_4, delta_4);

    // Matrix is 6 x 12
    strain_relationship.resize(B1.rows(), B1.cols() * 4);
    strain_relationship << B1, B2, B3, B4;
    if (V != 0.f) {
        strain_relationship /= (6 * V);
    } else {
        strain_relationship /= (6);
    }
    return strain_relationship;
}
auto solvers::fem::LinearElastic::ComputeTetrahedralElementVolume(const Vector3r &shape_one, const Vector3r &shape_two,
                                                                  const Vector3r &shape_three,
                                                                  const Vector3r &shape_four) -> Real {
    const Real x1 = shape_one.x();
    const Real y1 = shape_one.y();
    const Real z1 = shape_one.z();

    const Real x2 = shape_two.x();
    const Real y2 = shape_two.y();
    const Real z2 = shape_two.z();

    const Real x3 = shape_three.x();
    const Real y3 = shape_three.y();
    const Real z3 = shape_three.z();

    const Real x4 = shape_four.x();
    const Real y4 = shape_four.y();
    const Real z4 = shape_four.z();

    Matrix4r V;
    V.row(0) << 1, x1, y1, z1;
    V.row(1) << 1, x2, y2, z2;
    V.row(2) << 1, x3, y3, z3;
    V.row(3) << 1, x4, y4, z4;

    return V.determinant() / 6;
}
