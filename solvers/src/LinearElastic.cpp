// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <solvers/FEM/LinearElastic.h>

auto solvers::fem::LinearElastic::SolveWithIntegrator() -> void {}
auto solvers::fem::LinearElastic::SolveStatic() -> void {}
auto solvers::fem::LinearElastic::AssembleGlobalStiffness() -> void {}
auto solvers::fem::LinearElastic::AssembleElementStiffness() -> void {}

auto solvers::fem::LinearElastic::AssemblePlaneStresses(const MatrixXr &sigma) -> MatrixXr { return MatrixXr(); }
auto solvers::fem::LinearElastic::ComputeElementStress(const VectorXr &nodal_displacement) -> MatrixXr {
    return MatrixXr();
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
