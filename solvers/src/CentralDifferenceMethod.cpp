// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <solvers/integrators/CentralDifferenceMethod.h>
#include <utility>


solvers::integrators::CentralDifferenceMethod::CentralDifferenceMethod(Real dt, Real point_mass, MatrixXr stiffness,
                                                                       const VectorXr &initial_displacements,
                                                                       const VectorXr &initial_forces)
    : dt(dt), stiffness_(std::move(stiffness)) {
    SetMassMatrix(point_mass);
    SetDamping(0.5f, 0.5f);
    SetIntegrationConstants(dt);
    SetEffectiveMassMatrix();
    SetMovementVectors(initial_displacements, initial_forces, mass_matrix_);
    SetLastPosition(initial_displacements);
}

solvers::integrators::CentralDifferenceMethod::CentralDifferenceMethod(Real dt, const SparseMatrixXr &mass_matrix,
                                                                       MatrixXr stiffness,
                                                                       const VectorXr &initial_displacements,
                                                                       const VectorXr &initial_forces)
    : dt(dt), stiffness_(std::move(stiffness)) {
    SetMassMatrix(mass_matrix);
    SetDamping(0.5f, 0.5f);
    SetIntegrationConstants(dt);
    SetEffectiveMassMatrix();
    SetMovementVectors(initial_displacements, initial_forces, mass_matrix_);
    SetLastPosition(initial_displacements);
}

void solvers::integrators::CentralDifferenceMethod::SetDamping(Real mu, Real lambda) {
    ComputeRayleighDamping(stiffness_, mass_matrix_, mu, lambda, 0, damping_);
}

void solvers::integrators::CentralDifferenceMethod::SetMassMatrix(Real point_mass) {
    mass_matrix_.resize(stiffness_.rows(), stiffness_.cols());
    mass_matrix_.setIdentity();
    mass_matrix_ *= point_mass;
}

void solvers::integrators::CentralDifferenceMethod::SetMassMatrix(const SparseMatrixXr &m) { mass_matrix_ = m; }

void solvers::integrators::CentralDifferenceMethod::SetIntegrationConstants(Real dt) noexcept {
    a0 = 1.f / (std::powf(dt, 2));
    a1 = 1.f / (2.f * dt);
    a2 = 2.f * a0;
    a3 = 1.f / a2;
}

void solvers::integrators::CentralDifferenceMethod::Solve(const VectorXr &forces, VectorXr &displacements) {
    const VectorXr effective_load = ComputeEffectiveLoad(displacements, forces);

    const VectorXr next_displacement = effective_mass_matrix_ * effective_load;

    acceleration_ = a0 * (previous_position - 2 * displacements + next_displacement);
    velocity_ = a1 * ((-1 * previous_position) + next_displacement);
    previous_position = displacements;
    displacements = next_displacement;
}

void solvers::integrators::CentralDifferenceMethod::SetEffectiveMassMatrix() {
    effective_mass_matrix_ = a0 * mass_matrix_ + a1 * damping_;
    Eigen::SimplicialLDLT<SparseMatrixXr> solver;
    solver.compute(effective_mass_matrix_);
    SparseMatrixXr I(effective_mass_matrix_.rows(), effective_mass_matrix_.cols());
    I.setIdentity();
    effective_mass_matrix_ = solver.solve(I);
}

void solvers::integrators::CentralDifferenceMethod::SetLastPosition(const VectorXr &positions) {
    previous_position = positions - dt * velocity_ + a3 * acceleration_;
}

void solvers::integrators::CentralDifferenceMethod::SetMovementVectors(const VectorXr &positions,
                                                                       const VectorXr &forces,
                                                                       const MatrixXr &mass_matrix) {
    velocity_.resize(positions.rows());
    velocity_.setZero();
    acceleration_.resize(positions.rows());
    acceleration_ = mass_matrix.inverse() * forces;
}

auto solvers::integrators::CentralDifferenceMethod::ComputeEffectiveLoad(const VectorXr &displacements,
                                                                         const VectorXr &forces) const -> VectorXr {
    return forces - (stiffness_ - a2 * mass_matrix_) * displacements -
           (a0 * mass_matrix_ - a1 * damping_) * previous_position;
}

auto solvers::integrators::CentralDifferenceMethod::ComputeRayleighDamping(const MatrixXr &stiffness,
                                                                           const MatrixXr &mass, Real mu, Real lambda,
                                                                           Real mod, MatrixXr &out) -> void {
    out = mod * (mu * mass + lambda * stiffness);
}
