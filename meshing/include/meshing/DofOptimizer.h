// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_DOFOPTIMIZER_H
#define NEON_DOFOPTIMIZER_H

#include <meshing/Mesh.h>
#include <utilities/math/LinearAlgebra.h>
#include <vector>

namespace meshing {
    enum Axis {
        X = 0,
        Y,
        Z,
        ALL,
    };

    constexpr bool kMaxNodes = true;
    constexpr bool kMinNodes = false;

    void DofOptimizeUniaxial(Axis axis, bool max, const std::shared_ptr<meshing::Mesh> &mesh,
                             std::vector<unsigned int> &interior_nodes, std::vector<unsigned int> &force_nodes,
                             std::vector<unsigned int> &fixed_nodes);
    void DofOptimizeMultiAxial(const std::vector<Axis> &axes, const std::vector<bool> &min_max,
                               const std::vector<Axis> &fixed_axes, const std::vector<bool> &min_max_fixed,
                               const std::shared_ptr<meshing::Mesh> &mesh, std::vector<unsigned int> &active_nodes,
                               std::vector<unsigned int> &force_nodes, std::vector<unsigned int> &fixed_nodes);
    void DofOptimizeClear(Axis axis, bool max, const std::shared_ptr<meshing::Mesh> &mesh,
                          std::vector<unsigned int> &active_nodes, std::vector<unsigned int> &force_nodes,
                          std::vector<unsigned int> &fixed_nodes);
    void FindSurfaceNodes(Axis axis, bool max, const std::shared_ptr<meshing::Mesh> &mesh,
                          std::vector<unsigned int> &nodes);
    void FindSurfaceNodes(std::vector<Axis> axes, bool max, const std::shared_ptr<meshing::Mesh> &mesh,
                          std::vector<unsigned int> &nodes);
    void FindInteriorNodes(const std::vector<unsigned int> &excluded, std::size_t total_nodes,
                           std::vector<unsigned int> &interior_nodes);
}// namespace meshing


#endif//NEON_DOFOPTIMIZER_H
