// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_BOUNDARYCONDITION_H
#define NEON_BOUNDARYCONDITION_H

#include <set>
#include <utilities/math/LinearAlgebra.h>
#include <vector>

namespace solvers::boundary_conditions {
    struct BoundaryCondition {
        unsigned int node;
        Vector3r force;
    };

    using BoundaryConditions = std::vector<BoundaryCondition>;

    auto Nodes(const BoundaryConditions &bcs) -> std::vector<unsigned int>;
    auto FindYAxisBottomNodes(const MatrixXr &V) -> std::vector<unsigned int>;
    auto FindYAxisTopNodes(const MatrixXr &V) -> std::vector<unsigned int>;
    auto SelectNodes(const std::vector<unsigned int> &ignored, const MatrixXr &V) -> std::vector<unsigned int>;
    auto ApplyForceToBoundaryConditions(const std::vector<unsigned int> &indices, const Vector3r &force)
            -> BoundaryConditions;
}// namespace solvers::boundary_conditions
#endif//NEON_BOUNDARYCONDITION_H
