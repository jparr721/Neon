// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_ADJACENCYLIST_H
#define NEON_ADJACENCYLIST_H

#include <math/LinearAlgebra.h>

namespace meshing::graphs {
    class AdjacencyList {
    public:
        AdjacencyList(MatrixXr &V, MatrixXi &F);
        void ComputeAdjacencies();

        void WriteDataset();

        // Setters ================
        void SetParameters(const MatrixXr &params);

    private:
        // Adjacencies are arranged by the node number which corresponds to the
        MatrixXr adjacencies_;
        MatrixXr parameters_;
    };
}// namespace meshing::graphs

#endif//NEON_ADJACENCYLIST_H
