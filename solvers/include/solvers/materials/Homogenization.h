// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_HOMOGENIZATION_H
#define NEON_HOMOGENIZATION_H

#include <solvers/materials/Material.h>
#include <utilities/math/LinearAlgebra.h>

namespace solvers::materials {
    class Homogenization {
        using MatrixXi = MatrixX<int>;
        using VectorXi = VectorX<int>;

    public:
        struct MaterialCoefficients {
            Real E_11 = 0;
            Real E_22 = 0;
            Real E_33 = 0;

            Real G_23 = 0;
            Real G_31 = 0;
            Real G_12 = 0;

            Real v_21 = 0;
            Real v_31 = 0;
            Real v_12 = 0;
            Real v_32 = 0;
            Real v_13 = 0;
            Real v_23 = 0;

            auto Vector() const -> VectorXr {
                VectorXr v(12);
                v << E_11, E_22, E_33, G_23, G_31, G_12, v_21, v_31, v_12, v_32, v_13, v_23;

                return v;
            }
        };
        Homogenization(Tensor3r implicit_surface, const Material &material_1);
        Homogenization(Tensor3r implicit_surface, const Material &material_1, const Material &material_2);
        virtual ~Homogenization() = default;

        auto Stiffness() const -> Matrix6r { return constitutive_tensor_; }
        auto CoefficientVector() const -> VectorXr { return coefficients_.Vector(); }

        /// <summary>
        /// Solves the integral over the volume of the voxel for the difference of
        /// the macro and micro scale strain tensors.
        /// </summary>
        auto Solve() -> void;

        /// <summary>
        /// Creates the matrix S by inverting the stiffness constutive tensor C and
        /// gets the 6x6 compliance matrix which contains our material coefficients.
        /// </summary>
        /// <returns></returns>
        auto ComputeMaterialCoefficients() -> void;

        /// <summary>
        /// Computes the finite element approximation of the stiffness and load
        /// matrices
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        auto ComputeHexahedron(Real a, Real b, Real c) -> std::array<MatrixXr, 4>;

        auto ComputeElementDegreesOfFreedom(unsigned int n_elements) -> MatrixXi;
        auto ComputeUniqueNodes(unsigned int n_elements) -> Tensor3i;
        auto ComputeUniqueDegreesOfFreedom(const MatrixXi &element_degrees_of_freedom, const Tensor3i &unique_nodes)
                -> MatrixXi;

        auto AssembleStiffnessMatrix(unsigned int n_degrees_of_freedom, const MatrixXi &unique_degrees_of_freedom,
                                     const MatrixXr &ke_lambda, const MatrixXr &ke_mu) -> SparseMatrixXr;
        auto AssembleLoadMatrix(unsigned int n_elements, unsigned int n_degrees_of_freedom,
                                const MatrixXi &unique_degrees_of_freedom, const MatrixXr &fe_lambda,
                                const MatrixXr &fe_mu) -> SparseMatrixXr;

        /// \brief Solves the finite element method with conjugate gradient with incomplete
        /// cholesky preconditioner.
        /// @param n_degrees_of_freedom Total degrees of freedom for all
        /// nodes
        /// @param stiffness The stiffness matrix K
        /// @param load The load matrix F
        /// @param unique_degrees_of_freedom The degrees of freedom for
        /// non-void regions
        /// @returns Nodal displacement matrix Chi (X_e)
        auto ComputeDisplacement(unsigned int n_degrees_of_freedom, const MatrixXr &stiffness, const MatrixXr &load,
                                 const MatrixXi &unique_degrees_of_freedom) -> MatrixXr;
        auto ComputeUnitStrainParameters(unsigned int n_elements, const std::array<MatrixXr, 4> &hexahedron)
                -> Tensor3r;

        auto Voxel() const -> const Tensor3r & { return voxel_; }

    private:
        bool is_one_material_ = false;
        bool is_homogenized_ = false;

        unsigned int cell_len_x_ = 0;
        unsigned int cell_len_y_ = 0;
        unsigned int cell_len_z_ = 0;

        Real homogenized_E_ = 0;
        Real homogenized_v_ = 0;

        Matrix6r constitutive_tensor_;

        Tensor3r lambda_;
        Tensor3r mu_;

        Tensor3r voxel_;

        Material primary_material_;

        MaterialCoefficients coefficients_;

        // Constitutive Tensor Collection
        auto AssembleConstitutiveTensor(const MatrixXi &unique_degrees_of_freedom, const MatrixXr &ke_lambda,
                                        const MatrixXr &ke_mu, const MatrixXr &displacement,
                                        const Tensor3r &unit_strain_parameter) -> void;
    };
}// namespace solvers::materials


#endif//NEON_HOMOGENIZATION_H
