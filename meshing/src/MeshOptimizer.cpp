// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <igl/colon.h>
#include <meshing/MeshOptimizer.h>
#include <utilities/runtime/NeonAssert.h>
#include <utilities/runtime/NeonLog.h>

void meshing::optimizer::ComputeNormals(const MatrixXr &V, const MatrixXi &F, const Vector3r &D, MatrixXr &N) {
    NEON_ASSERT_ERROR(F.cols() == 3, "Faces must be non-volumetric");
    N.resize(F.rows(), F.cols());

    auto rows = F.rows();
    // Parallelism breaks down when the meshes are relatively small
#pragma omp parallel for if (rows > 10000)
    for (int row = 0; row < rows; ++row) {
        const int f1 = F(row, 0);
        const int f2 = F(row, 1);
        const int f3 = F(row, 2);

        // (AB) B - A
        const RowVector3r u = V.row(f2) - V.row(f1);

        // (BC) C - B
        const RowVector3r v = V.row(f3) - V.row(f1);

        // AB X BC
        const Vector3r norm = u.cross(v);
        N.row(row) = norm;
        const Real r = norm.norm();

        // r is a degenerate normal, so we use the value of D (the default)
        if (r == 0) {
            N.row(row) = D;
        } else {
            N.row(row) /= r;
        }
    }
}

void meshing::optimizer::ComputeAdjacencies(const MatrixXi &F, SparseMatrixXi &A) {
    using triplet = Eigen::Triplet<int>;
    std::vector<triplet> triplets;
    triplets.reserve(F.size() * 2);

    // Loop over all faces and assign their position for each individual position
    for (int row = 0; row < F.rows(); ++row) {
        // Loop over this row getting all combinations
        for (int c1 = 0; c1 < F.cols(); ++c1) {
            // Nest another loop for all pairs for this row. Start at 1 + c1 to avoid adding loops
            for (int c2 = c1 + 1; c2 < F.cols(); ++c2) {
                // Now, get the index from the faces bi-direcitonally.
                const int index1 = F(row, c1);
                const int index2 = F(row, c2);
                triplets.emplace_back(index1, index2, 1);
                triplets.emplace_back(index2, index1, 1);
            }
        }
    }

    const int n = F.maxCoeff() + 1;
    A.resize(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());

    // Since the sparse matrix class merges duplicates, we want to merge nonzero elements into ones.
    for (int i = 0; i < A.outerSize(); ++i) {
        // Spin off an iterator over just nonzero elements
        for (typename SparseMatrixXi::InnerIterator it(A, i); it; ++it) {
            // Make sure no zeroes snuck through, that would be quite bad
            NEON_ASSERT_ERROR(it.value() != 0, "Somehow a zero snuck in here, which is bad. Look into this!");

            // Set the value to one, since this is just an adjacency list we just want it to be a 1 to signify an edge.
            A.coeffRef(it.row(), it.col()) = 1;
        }
    }
}

void meshing::optimizer::ComputeEdges(const MatrixXr &V, const MatrixXi &F, MatrixXi &E) {
    SparseMatrixXi A;
    ComputeAdjacencies(F, A);

    // Number of non zeros should be twice the number of edges (since each nz is one side of the edge)
    NEON_ASSERT_ERROR(A.nonZeros() % 2 == 0, "Number of nonzeros should be evenly divisible by 2.", "\n",
                      "This is kinda bad, is F empty?");

    E.resize(A.nonZeros() / 2, 2);

    // The current row of E we are indexing into.
    int row = 0;

    // Iterate the outer size, which is the number of terms.
    for (int i = 0; i < A.outerSize(); ++i) {
        // Spin up an iterator for each block
        for (typename SparseMatrixXi::InnerIterator it(A, i); it; ++it) {
            // Now, assign the edge only in one direction. This works because the row is the index, but the
            // column maps to the same index, so we prevent a loop by only assigning going forward, not referencing
            // other indices from before it.row()
            if (it.row() < it.col()) {
                // Left side, the row we are currently on
                E(row, 0) = it.row();

                // Right side, the row we are getting the term for.
                E(row, 1) = it.col();

                ++row;
            }
        }
    }

    NEON_ASSERT_ERROR(row == E.rows(), "Computed adjacency matrix was not symmetric, this is a bug!! Fix me!", "\n",
                      "DEBUG INFO: \n", "F: \n", F, "\nA: \n", A.toDense());
}

void meshing::optimizer::ComputeEdgeLengths(const MatrixXr &V, const MatrixXi &F, MatrixXr &L) {
    NEON_ASSERT_ERROR(F.cols() == 3, "Triangular meshes only!");

    const int rows = F.rows();
    L.resize(rows, 3);
#pragma omp parallel for if (rows > 10000)
    for (int row = 0; row < rows; ++row) {
        const int A_idx = F(row, 0);
        const int B_idx = F(row, 1);
        const int C_idx = F(row, 2);

        // BC Edge
        L(row, 0) = (V.row(B_idx) - V.row(C_idx)).norm();

        // A C Edge
        L(row, 1) = (V.row(A_idx) - V.row(C_idx)).norm();

        // A B Edge
        L(row, 2) = (V.row(A_idx) - V.row(B_idx)).norm();
    }
}

void meshing::optimizer::ComputeTriangleSquaredArea(const MatrixXr &L, VectorXr &A) {
    const int rows = L.rows();
    A.resize(rows);
#pragma omp parallel for if (rows > 10000)
    for (int row = 0; row < rows; ++row) {
        const Real a = L(row, 0);
        const Real b = L(row, 1);
        const Real c = L(row, 2);
        // Semi-perimeter = sum of side lengths divided by 2.
        const Real s = L.row(row).sum() / 2;
        A(row) = s * (s - a) * (s - b) * (s - c);
    }
}

void meshing::optimizer::CollapseSmallTriangles(Real min_area, const MatrixXr &V, const MatrixXi &F, MatrixXr &VV,
                                                MatrixXi &FF) {
    NEON_ASSERT_ERROR(F.cols() == 3, "Can only edge collapse on triplet surface mesh, not quads.");

    int n_face_collapses = 0;
    int n_edge_collapses = 0;

    // Compute the edge lengths for each triangle
    MatrixXr L;
    ComputeEdgeLengths(V, F, L);

    // Compute the squared area for each triangle.
    VectorXr A;
    ComputeTriangleSquaredArea(L, A);

    VectorXi F_intermediate;
    // Pre-fill with all the face indices
    F_intermediate = igl::colon<int>(0, F.rows() - 1);
    for (int face = 0; face < F.rows(); ++face) {
        // If the triangle is less than the min area, collapse. By getting the max length edge and the
        // min length edge; this gives us the indices that we need to change.
        if (A(face) < min_area) {
            // Get the shortest edge
            Real min_length = 0;
            int min_length_index = -1;

            for (int edge = 0; edge < F.cols(); ++edge) {
                // Since lengths can theoretically be pretty small, we don't want to constrain by an arbitrarily set
                // starting value, instead we just set it to the first thing we see. And then compare from there
                if (min_length_index == -1 || L(face, edge) < min_length) {
                    min_length_index = edge;
                    min_length = L(face, edge);
                }
            }

            // Get the longest edge
            Real max_length = 0;
            int max_length_index = -1;

            for (int edge = 0; edge < F.cols(); ++edge) {
                // Since lengths can theoretically be pretty big, we don't want to constrain by an arbitrarily set
                // starting value, instead we just set it to the first thing we see. And then compare from there
                if (max_length_index == -1 || L(face, edge) < max_length) {
                    max_length_index = edge;
                    max_length = L(face, edge);
                }
            }

            // If the two selected indices are the same, we need to arbitrarily pick another index
            if (max_length_index == min_length_index) { max_length_index = (min_length_index + 1) % 3; }

            // Collapse the minimum edge and place it in our intermediate face matrix.
            // Re-assign the min length index value to the adjacent side of the max length index
            int j = ((min_length_index + 1) % 3) == max_length_index ? (min_length_index + 2) % 3
                                                                     : (min_length_index + 1) % 3;
            int i = max_length_index;

            // Now, set the intermediate mapping vector with the value of the max face. This will "break" the face
            // so we can prune it later.
            F_intermediate(F(face, i)) = F_intermediate(F(face, j));
            ++n_edge_collapses;
        }
    }

    MatrixXi F_reindexed;
    F_reindexed = F;
    for (int row = 0; row < F_reindexed.rows(); ++row) {
        for (int col = 0; col < F_reindexed.cols(); ++col) {
            // Assign the reindexed faces to the updated position from the intermediate vector.
            // This will purposefully result in duplicate faces showing up in the reindexed vector, this will
            // be used to signify that the face should be removed since it's degenerate (since it was collapsed
            // to a regular ol' line).
            F_reindexed(row, col) = F_intermediate(F_reindexed(row, col));
        }
    }

    // Determine whether or not to keep faces in the reindex matrix by checking for duplicate faces and, if found,
    // skipping it and removing the face entirely
    // Resize to the same dims as the reindexed vector. We will do a conservative resize when we've pruned the
    // broken faces
    FF.resizeLike(F_reindexed);

    int ffi = 0;

    for (int face = 0; face < F_reindexed.rows(); ++face) {
        bool is_face_collapsed = false;

        // Check for duplicate indices on each face
        for (int i = 0; i < F_reindexed.cols(); ++i) {
            for (int j = i + 1; j < F_reindexed.cols(); ++j) {
                // If we find a duplicate, we skip this face
                if (F_reindexed(face, i) == F_reindexed(face, j)) {
                    is_face_collapsed = true;
                    ++n_face_collapses;
                }
            }
        }

        // If the face is not a degenerate triangle, move on to add it to the new faces.
        if (!is_face_collapsed) {
            // This index is independent of the reindexed index for obvious reasons.
            FF.row(ffi) = F_reindexed.row(face);
            ++ffi;
        }
    }

    // Clip the bottom rows from the FF matrix of new faces.
    FF.conservativeResize(ffi, FF.cols());

    if (n_edge_collapses == 0) {
        NEON_ASSERT_ERROR(n_face_collapses == 0,
                          "Faces collapsed without an edge collapse. The mesh contains degenerate triangles!");
        return;
    }

    // Recurse until no small triangles remain.
    MatrixXi FC = FF;
    return CollapseSmallTriangles(min_area, V, FC, VV, FF);
}
