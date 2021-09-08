// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#include <algorithm>
#include <solvers/utilities/BoundaryCondition.h>

void solvers::boundary_conditions::LoadBoundaryConditions(
        const Vector3r &force, const std::shared_ptr<meshing::Mesh> &mesh, const std::vector<unsigned int> &force_nodes,
        const std::vector<unsigned int> &active_nodes,
        solvers::boundary_conditions::BoundaryConditions &boundary_conditions) {
    for (const auto &index : force_nodes) { boundary_conditions.emplace_back(BoundaryCondition{index, force}); }

    for (const auto &index : active_nodes) {
        boundary_conditions.emplace_back(BoundaryCondition{index, Vector3r::Zero()});
    }
}
