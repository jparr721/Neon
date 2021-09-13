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
    struct DynamicSolverDataset {
        // Track which entry we've filled.
        unsigned int current_input_entry = 0;
        unsigned int current_target_entry = 0;

        // Rank-4 Tensor
        Eigen::Tensor<Real, 5> input;
        Eigen::Tensor<Real, 5> target;

        DynamicSolverDataset(unsigned int shape, int entries);
        void AddInputEntry(int layer, int feature, const MatrixXr &data);
        void AddTargetEntry(int layer, int feature, const MatrixXr &data);

        void Sides(std::vector<MatrixXr> &bitmasks);
        void Tops(std::vector<MatrixXr> &bitmasks);
        void Fronts(std::vector<MatrixXr> &bitmasks);

        auto Shape() const -> const unsigned int;
        auto Entries() const -> const unsigned int;
    };

    class DynamicSolverMask {
    public:
        DynamicSolverMask(unsigned int shape, int entries);
        ~DynamicSolverMask();

        void GenerateDataset(const Vector3r &force, Real mass, Real dt, Real E, Real v, Real G);

    private:
        std::fstream input_file_;
        std::fstream target_file_;

        DynamicSolverDataset dataset_;
        std::shared_ptr<meshing::Mesh> mesh_;
        std::unique_ptr<solvers::fem::LinearElastic> solver_;
        std::unique_ptr<solvers::integrators::CentralDifferenceMethod> integrator_;

        void Save();
        void Solve(MatrixXr &displacements, MatrixXr &stresses);

        void VectorToOrientedMatrix(const VectorXr &v, MatrixXr &m);
        void SetZerosFromBoundaryConditions(const VectorXr &v,
                                            const solvers::boundary_conditions::BoundaryConditions &boundary_conditions,
                                            MatrixXr &m);
    };
}// namespace datasets

#endif//NEON_SOLVERMASK_H
