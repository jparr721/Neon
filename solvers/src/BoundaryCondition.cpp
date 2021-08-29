// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#include <solvers/helpers/BoundaryCondition.h>

auto solvers::helpers::ApplyForceToBoundaryConditions(const std::set<unsigned int> &indices, const Vector3r &force)
        -> BoundaryConditions {
    BoundaryConditions conditions;
    for (const auto &index : indices) { conditions.emplace_back(BoundaryCondition{index, force}); }
    return conditions;
}
auto solvers::helpers::FindYAxisBottomNodes(const MatrixXr &V) -> std::set<unsigned int> {
    Real min_y = 1e10;
    std::set<unsigned int> indices;
    for (int row = 0; row < V.rows(); ++row) {
        const Vector3r triplet = V.row(row);
        min_y = std::fmin(triplet.y(), min_y);
    }

    for (int row = 0; row < V.rows(); ++row) {
        const Vector3r triplet = V.row(row);
        if (triplet.y() <= min_y) { indices.insert(row); }
    }

    return indices;
}
auto solvers::helpers::FindYAxisTopNodes(const MatrixXr &V) -> std::set<unsigned int> {
    Real max_y = -1e10;
    std::set<unsigned int> indices;
    for (int row = 0; row < V.rows(); ++row) {
        const Vector3r triplet = V.row(row);
        max_y = std::fmax(triplet.y(), max_y);
    }

    for (int row = 0; row < V.rows(); ++row) {
        const Vector3r triplet = V.row(row);
        if (triplet.y() >= max_y) { indices.insert(row); }
    }

    return indices;
}
auto solvers::helpers::SelectNodes(const std::set<unsigned int> &ignored, const MatrixXr &V) -> std::set<unsigned int> {
    std::set<unsigned int> indices;
    for (int row = 0; row < V.rows(); ++row) {
        // If the value isn't found, add it to our indices.
        if (ignored.find(row) == ignored.end()) { indices.insert(row); }
    }

    return indices;
}
