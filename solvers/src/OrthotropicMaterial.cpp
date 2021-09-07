// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <solvers/materials/OrthotropicMaterial.h>
#include <utilities/runtime/NeonLog.h>
solvers::materials::OrthotropicMaterial::OrthotropicMaterial(const VectorXr &coefficients)
    : coefficients(coefficients) {
    E_x = coefficients(0);
    E_y = coefficients(1);
    E_z = coefficients(2);
    G_yz = coefficients(3);
    G_zx = coefficients(4);
    G_xy = coefficients(5);
    v_yx = coefficients(6);
    v_zx = coefficients(7);
    v_zy = coefficients(8);
    v_xy = coefficients(9);
    v_xz = coefficients(10);
    v_yz = coefficients(11);
}
solvers::materials::OrthotropicMaterial::OrthotropicMaterial(const Real E, const Real v) {
    const Real G = E / (2 * (1 + v));

    E_x = E;
    E_y = E;
    E_z = E;
    G_yz = G;
    G_zx = G;
    G_xy = G;
    v_yx = v;
    v_zx = v;
    v_zy = v;
    v_xy = v;
    v_xz = v;
    v_yz = v;

    coefficients.resize(12);
    coefficients << E_x, E_y, E_z, G_yz, G_zx, G_xy, v_yx, v_zx, v_zy, v_xy, v_xz, v_yz;
}

/// Returns the material coefficients for Hooke's Law in Stiffness Form.
/// \return Constitutive Matrix in Stiffness Form (6x6)
auto solvers::materials::OrthotropicMaterial::ConstitutiveMatrix() -> Matrix6r {
    const Real delta = (1 - v_xy * v_yx - v_yz * v_zy - v_zx * v_xz - 2 * (v_xy * v_yz * v_zx)) / (E_x * E_y * E_z);

    Matrix6r constitutive_matrix;
    constitutive_matrix.setZero();

    constitutive_matrix(0, 0) = (1 - v_yz * v_zy) / (E_y * E_z * delta);
    constitutive_matrix(0, 1) = (v_yx + v_zx * v_yz) / (E_y * E_z * delta);
    constitutive_matrix(0, 2) = (v_zx + v_yx * v_zy) / (E_y * E_z * delta);

    constitutive_matrix(1, 0) = (v_xy + v_xz * v_zy) / (E_z * E_x * delta);
    constitutive_matrix(1, 1) = (1 - v_zx * v_xz) / (E_z * E_x * delta);
    constitutive_matrix(1, 2) = (v_zy + v_zx * v_xy) / (E_z * E_x * delta);

    constitutive_matrix(2, 0) = (v_xz + v_xy * v_yz) / (E_x * E_y * delta);
    constitutive_matrix(2, 1) = (v_yz + v_xz * v_yx) / (E_x * E_y * delta);
    constitutive_matrix(2, 2) = (1 - v_xy * v_yx) / (E_x * E_y * delta);

    // NOTE: These parameters differ from the isotropic form, more details here if this causes problems:
    // https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_orthotropic.cfm
    constitutive_matrix(3, 3) = 2 * G_yz;
    constitutive_matrix(4, 4) = 2 * G_zx;
    constitutive_matrix(5, 5) = 2 * G_xy;

    return constitutive_matrix;
}
