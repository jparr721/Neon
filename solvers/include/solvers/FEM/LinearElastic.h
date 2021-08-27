//
// Created by jparr on 8/21/2021.
//

#ifndef NEON_LINEARELASTIC_H
#define NEON_LINEARELASTIC_H

#include <Eigen/Dense>
#include <memory>
#include <meshing/Mesh.h>
#include <solvers/helpers/BoundaryCondition.h>
#include <utilities/math/LinearAlgebra.h>

namespace solvers::fem {
    class LinearElastic {
    public:
        /// \brief Global stiffness matrix.
        MatrixXr K;

        /// \brief Per-node element stiffness matrix.
        MatrixXr K_e;

        /// \brief Global displacement vector.
        VectorXr U;

        /// \brief Nodal boundary conditions.
        helpers::BoundaryConditions boundary_conditions;

        auto SolveWithIntegrator() -> void;
        auto SolveStatic() -> void;

        /// \brief Assemble the global stiffness matrix by slicing together all of the
        /// element stiffness matrices.
        auto AssembleGlobalStiffness() -> void;

        /// \brief Assemble the element stiffness matrix by solving K_e = V * B^T * D * B.
        auto AssembleElementStiffness() -> void;

        auto AssembleConstitutiveMatrix() -> void;

        /// \brief Computes the plane stresses for the tetrahedral element.
        auto AssemblePlaneStresses(const MatrixXr &sigma) -> MatrixXr;

        auto ComputeElementStress(const VectorXr &nodal_displacement) -> MatrixXr;

        /// \brief Computes the shape function parameter from 6 dof points.
        auto ComputeShapeFunctionFromPoints(const Vector6r &points) -> Real;

        auto AssembleStrainRelationshipMatrix(const Vector3r &shape_one, const Vector3r &shape_two,
                                              const Vector3r &shape_three, const Vector3r &shape_four) -> MatrixXr;

    private:
        Real youngs_modulus_;
        Real poissons_ratio_;

        std::shared_ptr<meshing::Mesh> mesh_;

        Matrix6r constitutive_matrix_;

        auto ComputeTetrahedralElementVolume(const Vector3r &shape_one, const Vector3r &shape_two,
                                             const Vector3r &shape_three, const Vector3r &shape_four) -> Real;
    };

}// namespace solvers::fem
#endif//NEON_LINEARELASTIC_H
