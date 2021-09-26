// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <meshing/MeshOptimizer.h>
#include <utilities/runtime/NeonAssert.h>

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

void meshing::optimizer::EdgeCollapseSlivers(const Real min_theta, MatrixXr &V, MatrixXi &F, MatrixXi &E) {
    NEON_ASSERT_ERROR(F.cols() == 3, "Can only edge collapse on triplet surface mesh, not quads.");
    for (int row = 0; row < F.rows(); ++row) {
        const int fa = F(row, 0);
        const int fb = F(row, 1);
        const int fc = F(row, 2);

        const Vector3r a = V.row(fa);
        const Vector3r b = V.row(fb);
        const Vector3r c = V.row(fc);

        // Find the smallest angle.
        // A -> B
        const Real ab = utilities::math::ComputeAngle(a, b);

        // A -> C
        const Real ac = utilities::math::ComputeAngle(a, c);

        // B -> C
        const Real bc = utilities::math::ComputeAngle(b, c);

        const std::vector<Real> angles{ab, ac, bc};
        const Real min = *std::min_element(angles.begin(), angles.end());

        // Now check if it's smaller than our minimum theta.
        if (min > min_theta) { continue; }

        NEON_ASSERT_ERROR(utilities::math::IsApprox(min, ab, 0.001) || utilities::math::IsApprox(min, ac, 0.001) ||
                                  utilities::math::IsApprox(min, bc, 0.001),
                          "Something has gone terribly wrong: ", min);

        // The line segment between the two faces.
        std::pair<int, int> line;
        if (min == ab) {
            line = std::pair<int, int>{fa, fb};
        } else if (min == ac) {
            line = std::pair<int, int>{fa, fc};
        } else if (min == bc) {
            line = std::pair<int, int>{fb, fc};
        }

        // Find the face that is connected here
        // TODO(@jparr721) Is the found face unique?
        for (int r = 0; r < F.rows(); ++r) {
            if (r == row) { continue; }

            const int ffa = F(r, 0);
            const int ffb = F(r, 1);
            const int ffc = F(r, 2);

            const Vector3i ffr = F.row(r);

            // If both of the nodes are here, we get to work
            if (!(utilities::math::Contains(ffr, line.first) && utilities::math::Contains(ffr, line.second))) {
                continue;
            }

            // Both nodes are here, so it shares an edge with the thing we want to collapse.

            // Now, find the "skinny" angle where the min_theta is
        }
    }
}
