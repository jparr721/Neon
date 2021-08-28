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
        struct ElementStiffness {
            Matrix12r stiffness;
            Vector4i tetrahedral;
        };

    public:
        enum Type {
            kStatic = 0,
            kDynamic,
        };

        /// \brief Denotes the type of solver we're working with.
        Type type;

        /// \brief Global stiffness matrix.
        MatrixXr K;

        /// \brief Per-node element stiffness matrices for integrated solution.
        std::vector<ElementStiffness> K_e;

        /// \brief Per-node element stiffness matrix for static solution.
        MatrixXr K_e_static;

        /// \brief Global displacement vector.
        VectorXr U;

        /// \brief Element displacement vector.
        VectorXr U_e;

        /// \brief Global force vector.
        VectorXr F;

        /// \brief The local boundary force vector (for active dofs only)
        VectorXr F_e;

        /// \brief Nodal boundary conditions.
        helpers::BoundaryConditions boundary_conditions;

        LinearElastic(helpers::BoundaryConditions boundary_conditions, Real youngs_modulus, Real poissons_ratio,
                      std::shared_ptr<meshing::Mesh> mesh, Type type = Type::kDynamic);

        auto SolveWithIntegrator() -> void;
        auto SolveStatic() -> void;

        /// \brief Assemble the global stiffness matrix by slicing together all of the
        /// element stiffness matrices.
        auto AssembleGlobalStiffness() -> void;

        /// \brief Assemble the element stiffness matrix by solving K_e = V * B^T * D * B.
        auto AssembleElementStiffness() -> void;

        auto AssembleBoundaryForces() -> void;

        auto AssembleConstitutiveMatrix() -> void;

        /// \brief Computes the plane stresses for the tetrahedral element.
        auto AssemblePlaneStresses(const MatrixXr &sigmas) -> MatrixXr;

        auto ComputeElementStress() -> MatrixXr;

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
