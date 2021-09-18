// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_SOLVERMASK_H
#define NEON_SOLVERMASK_H

#include <fstream>
#include <memory>
#include <meshing/Mesh.h>
#include <solvers/FEM/LinearElastic.h>
#include <solvers/integrators/CentralDifferenceMethod.h>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

namespace datasets {
    // rename
    struct DynamicSolverDataset {
        static constexpr unsigned int kTensorRank = 4;
        static constexpr unsigned int kDatasetFeatures = 2;

        // Feature indices
        static constexpr unsigned int kFeatureScalarField = 0;
        static constexpr unsigned int kFeatureForces = 1;

        static constexpr unsigned int kDisplacements = 0;
        static constexpr unsigned int kVelocity = 1;

        static constexpr auto kInputFileName = "input";
        static constexpr auto kTargetFileName = "target";

        using Tensor = Eigen::Tensor<Real, kTensorRank>;

        // Track which entry we've filled.
        unsigned int current_input_entry = 0;
        unsigned int current_target_entry = 0;

        Tensor input;
        Tensor target;

        DynamicSolverDataset(unsigned int shape_x, unsigned int shape_y, unsigned int entries);
        void AddInputEntry(int feature, const MatrixXr &data);
        void AddTargetEntry(int feature, const MatrixXr &data);

        void Sides(std::vector<MatrixXr> &bitmasks);
        void Tops(std::vector<MatrixXr> &bitmasks);
        void Fronts(std::vector<MatrixXr> &bitmasks);

        [[nodiscard]] auto Shape() const -> const unsigned int;
        [[nodiscard]] auto Entries() const -> const unsigned int;
    };

    // rename
    class DynamicSolverMask {
    public:
        unsigned int mesh_shape;
        unsigned int n_entries;

        DynamicSolverMask(unsigned int shape, int entries);
        ~DynamicSolverMask();

        void GenerateDataset(const Vector3r &force, Real mass, Real dt, Real E, Real v, Real G);

    private:
        std::fstream input_file_;
        std::fstream target_file_;

        std::shared_ptr<meshing::Mesh> mesh_;
        std::unique_ptr<solvers::fem::LinearElastic> solver_;
        std::unique_ptr<solvers::integrators::CentralDifferenceMethod> integrator_;

        void Save();
        void Solve(MatrixXr &displacements, MatrixXr &stresses);

        void VectorToOrientedMatrix(const VectorXr &v, MatrixXr &m);
        void SetZerosFromBoundaryConditions(const VectorXr &v,
                                            const solvers::boundary_conditions::BoundaryConditions &boundary_conditions,
                                            MatrixXr &m);

        // Helpers
        void PruneZeros(const MatrixXr &scalar_field, MatrixXr &pruned);
    };
}// namespace datasets

#endif//NEON_SOLVERMASK_H
