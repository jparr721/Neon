// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_MESHOPTIMIZER_H
#define NEON_MESHOPTIMIZER_H

#include <unordered_map>
#include <utilities/math/LinearAlgebra.h>
#include <utility>

namespace meshing::optimizer {
    const Vector3r kNoNormal = Vector3r::Zero();

    /// Compute the normals from a given face set via the cross product
    /// \param V The list of vertex positions in reference space
    /// \param F The list of faces which map to the vertices
    /// \param D The default value in case the calculated norm is degenerate
    /// \param N The normal matrix
    void ComputeNormals(const MatrixXr &V, const MatrixXi &F, const Vector3r &D, MatrixXr &N);

    /// Computes the adjacency matrix from an input set of faces
    /// \param F The matrix of faces
    /// \param A The output sparse matrix
    void ComputeAdjacencies(const MatrixXi &F, SparseMatrixXi &A);

    /// Computes the list of unique edges from a given mesh.
    /// \param V The vertex positions
    /// \param F The faces shape F by 3 or 4
    /// \param E The output matrix of edges shape F by 2
    void ComputeEdges(const MatrixXr &V, const MatrixXi &F, MatrixXi &E);

    /// Compute the list of lengths of edges by taking the norm of each. This is significantly faster than the academic
    /// version of the problem where you may ordinarily take the pythagorean theorem of the distance between the two
    /// points. Square roots are _very_ costly computationally, so we instead take the norm which does the same thing
    /// except quicker.
    /// \param V The vertex positions
    /// \param F The nx3 matrix of edges
    /// \param L The lengths for each face (idx row = face #)
    void ComputeEdgeLengths(const MatrixXr &V, const MatrixXi &F, MatrixXr &L);

    /// Computes the squared area of every face in the mesh using the edge lengths to compute perimeter. Then, using
    /// the squared heron's formula, we compute the area of each triangle.
    /// \param L The lengths for each face (idx row = face #)
    /// \param A The area of each face (# rows of F x 1)
    void ComputeTriangleSquaredArea(const MatrixXr &L, VectorXr &A);

    /// Collapses the sliver elements and updates the vertices and faces. The heuristic for this algorithm is quite
    /// simple. We take our minimum area value, compute the area of all triangles, any areas that are too small are
    /// collapsed on the shortest edge (to prevent the triangle from collapsing weirdly).
    /// \param min_area The min value of theta for the triangle to perform the edge collapse
    /// \param V The matrix of vertex positions
    /// \param F The matrix of face positions with proper winding
    /// \param VV The output matrix of refined vertex positions
    /// \param FF The output matrix of refined faces
    void CollapseSmallTriangles(Real min_area, const MatrixXr &V, const MatrixXi &F, MatrixXr &VV, MatrixXi &FF);
}// namespace meshing::optimizer

#endif//NEON_MESHOPTIMIZER_H
