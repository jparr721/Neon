// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_CENTRALDIFFERENCEMETHOD_H
#define NEON_CENTRALDIFFERENCEMETHOD_H

#include <utilities/math/LinearAlgebra.h>

namespace solvers::integrators {
    class CentralDifferenceMethod {
    public:
        /**
        * \brief Integration constant one, a0 = 1 / dt^2
        */
        Real a0;

        /**
         * \brief Integration constant two, a1 = 1 / 2 * dt
         */
        Real a1;

        /**
         * \brief Integration constant three, a2 = 2 * a0
         */
        Real a2;

        /**
         * \brief Integration constant four, a3 = 1 / a2
         */
        Real a3;

        /**
         * \brief Change in time per interval
         */
        const Real dt;

        /**
         * \brief The system displacement from the previous time step
         */
        VectorXr previous_position;

        CentralDifferenceMethod(Real dt, Real point_mass, SparseMatrixXr stiffness,
                                const VectorXr &initial_displacements, const VectorXr &initial_forces);

        CentralDifferenceMethod(Real dt, const SparseMatrixXr &mass_matrix, SparseMatrixXr stiffness,
                                const VectorXr &initial_displacements, const VectorXr &initial_forces);

        // Setters
        void SetMassMatrix(Real point_mass);
        void SetMassMatrix(const SparseMatrixXr &m);
        void SetIntegrationConstants() noexcept;

        // Getters
        [[nodiscard]] auto NodalMass() -> Real { return mass_matrix_.coeff(0, 0); }

        /**
        \brief Calculates the explicit Central Difference Method integration
        equation given the local position and velocity vectors.
    
        \n It solves according to the following algorithm:
        \n foreach time step {
            \n 1. Calculate effective loads
            \n effective_load <- forces - (stiffness - a2 * mass_matrix) *
        current_displacement - (a0 * mass_matrix) * previous_position
    
            \n 2. Calculate Displacements at dt
            effective_mass_matrix * effective_load = next_displacement
        \n }
    
        \param displacements The new displacement value
        \param forces The generalized stacked force vector
        solving for the displacement
        **/
        void Solve(const VectorXr &forces, VectorXr &displacements);

        [[nodiscard]] auto Velocity() const -> VectorXr { return velocity_; }
        [[nodiscard]] auto Acceleration() const -> VectorXr { return acceleration_; }

        auto ComputeRayleighDamping(Real mu = 0.5, Real lambda = 0.5, Real mod = 0) -> void;

    private:
        const SparseMatrixXr stiffness_;

        SparseMatrixXr mass_matrix_;
        SparseMatrixXr mass_matrix_inverse_;

        SparseMatrixXr damping_;
        SparseMatrixXr effective_mass_matrix_;

        VectorXr velocity_;
        VectorXr acceleration_;

        void SetEffectiveMassMatrix();
        void SetLastPosition(const VectorXr &positions);
        void SetMovementVectors(const VectorXr &positions, const VectorXr &forces);

        auto ComputeEffectiveLoad(const VectorXr &displacements, const VectorXr &forces) const -> VectorXr;
    };
}// namespace solvers::integrators

#endif//NEON_CENTRALDIFFERENCEMETHOD_H
