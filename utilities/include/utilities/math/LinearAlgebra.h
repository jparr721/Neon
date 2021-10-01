// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <igl/list_to_matrix.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utilities/runtime/NeonAssert.h>
#include <vector>

#ifndef NEON_LINEARALGEBRA_H
#define NEON_LINEARALGEBRA_H

#ifdef NEON_USE_DOUBLE
using Real = double;
#else
using Real = float;
#endif

// Reals
// Dense Vector Types
using Vector2r = Eigen::Matrix<Real, 2, 1>;
using Vector3r = Eigen::Matrix<Real, 3, 1>;
using Vector4r = Eigen::Matrix<Real, 4, 1>;
using Vector6r = Eigen::Matrix<Real, 6, 1>;
using Vector12r = Eigen::Matrix<Real, 12, 1>;
using VectorXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

// Sparse Vector Types
using SparseVectorXr = Eigen::SparseVector<Real>;
template<typename T>
using SparseVectorX = Eigen::SparseVector<T>;

// Dense Row Vector Types
using RowVector3r = Eigen::Matrix<Real, 1, 3>;
using RowVectorXr = Eigen::Matrix<Real, 1, Eigen::Dynamic>;

// Dense Matrix Types
using Matrix2r = Eigen::Matrix<Real, 2, 2>;
using Matrix3r = Eigen::Matrix<Real, 3, 3>;
using Matrix4r = Eigen::Matrix<Real, 4, 4>;
using Matrix6r = Eigen::Matrix<Real, 6, 6>;
using Matrix36r = Eigen::Matrix<Real, 3, 6>;
using Matrix12r = Eigen::Matrix<Real, 12, 12>;
using MatrixXr = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

// Sparse MatrixTypes
using SparseMatrixXr = Eigen::SparseMatrix<Real>;

// Integers
// Dense Vector Types
using Vector2i = Eigen::Matrix<int, 2, 1>;
using Vector3i = Eigen::Matrix<int, 3, 1>;
using Vector4i = Eigen::Matrix<int, 4, 1>;
using Vector6i = Eigen::Matrix<int, 6, 1>;
using Vector12i = Eigen::Matrix<int, 12, 1>;
using VectorXi = Eigen::Matrix<int, Eigen::Dynamic, 1>;

// Dense Row Vector Types
using RowVector3i = Eigen::Matrix<int, 1, 3>;
using RowVectorXi = Eigen::Matrix<int, 1, Eigen::Dynamic>;

// Dense Matrix Types
using Matrix2i = Eigen::Matrix<int, 2, 2>;
using Matrix3i = Eigen::Matrix<int, 3, 3>;
using Matrix4i = Eigen::Matrix<int, 4, 4>;
using Matrix6i = Eigen::Matrix<int, 6, 6>;
using Matrix36i = Eigen::Matrix<int, 3, 6>;
using Matrix12i = Eigen::Matrix<int, 12, 12>;
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

// Sparse MatrixTypes
using SparseMatrixXi = Eigen::SparseMatrix<int>;

template<typename T>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template<typename T>
using Vector2 = Eigen::Matrix<T, 2, 1>;
template<typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>;
template<typename T>
using Vector4 = Eigen::Matrix<T, 4, 1>;
template<typename T>
using Vector6 = Eigen::Matrix<T, 6, 1>;
template<typename T>
using Vector12 = Eigen::Matrix<T, 12, 1>;

template<typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template<typename T>
using Matrix2 = Eigen::Matrix<T, 2, 2>;
template<typename T>
using Matrix3 = Eigen::Matrix<T, 3, 3>;
template<typename T>
using Matrix4 = Eigen::Matrix<T, 4, 4>;


namespace utilities::math {
    // Static Variables ===============================
    constexpr Real kPi = static_cast<Real>(3.14159265);

    // Functions ======================================

    auto LinSpace(Real start, Real stop, unsigned int num) -> VectorXr;

    template<typename Derived>
    inline auto Shape(const Eigen::PlainObjectBase<Derived> &in) -> Vector2<int> {
        return Vector2<int>(in.rows(), in.cols());
    }

    template<typename T>
    inline auto MatrixToVector(const MatrixX<T> &in) -> VectorX<T> {
        const T *data = in.data();
        const auto shape = in.rows() * in.cols();
        return VectorX<T>(Eigen::Map<const VectorX<T>>(data, shape));
    }

    template<typename T>
    inline auto VectorToMatrix(const VectorX<T> &in, int rows, int cols) -> MatrixX<T> {
        const T *data = in.data();
        return MatrixX<T>(Eigen::Map<const MatrixX<T>>(data, rows, cols));
    }

    template<typename T>
    inline auto IndexVectorByMatrix(const VectorX<T> &in, const MatrixX<T> &indices) -> MatrixX<T> {
        MatrixX<T> output(indices.rows(), indices.cols());

        for (int row = 0; row < indices.rows(); ++row) {
            for (int col = 0; col < indices.cols(); ++col) { output(row, col) = in(indices(row, col)); }
        }

        return output;
    }

    template<typename T>
    inline auto ReArrange(const MatrixX<T> &in, const VectorX<int> &indices) -> MatrixX<T> {
        NEON_ASSERT_ERROR(indices.maxCoeff() < in.rows(), "Index out of bounds");

        MatrixX<T> output(indices.rows(), in.cols());
        for (int r = 0; r < indices.rows(); ++r) {
            const T row = indices(r);
            output.row(r) = in.row(row);
        }

        return output;
    }

    template<typename T>
    inline auto IndexMatrixByMatrix(const MatrixX<T> &in, const MatrixX<int> &indices) -> MatrixX<T> {
        std::vector<MatrixX<T>> h_stack;

        for (int row = 0; row < indices.rows(); ++row) {
            const VectorX<int> index_row = indices.row(row);
            const MatrixX<T> m = ReArrange(in, index_row);
            h_stack.emplace_back(m);
        }

        return HStack(h_stack);
    }

    template<typename T>
    inline auto IndexMatrixByMatrix(const MatrixX<T> &in, const MatrixX<int> &indices, const int col) -> MatrixX<T> {
        const MatrixX<T> stack = IndexMatrixByMatrix(in, indices);

        // Get the column we're slicing from
        const VectorX<T> column = stack.col(col);

        // Then, extract by segment into an STL container
        std::vector<MatrixX<T>> h_stack;
        const unsigned int stride = indices.cols();

        for (int i = 0; i < column.rows(); i += stride) {
            const MatrixX<T> v = column.segment(i, stride).transpose();
            h_stack.emplace_back(v);
        }

        // Return everything stacked out as a matrix
        return HStack(h_stack);
    }

    template<typename T>
    inline auto ToTriplets(const VectorX<int> &i, const VectorX<int> &j, const VectorX<T> &data)
            -> std::vector<Eigen::Triplet<T>> {
        NEON_ASSERT_ERROR(Shape(i) == Shape(j) && Shape(i) == Shape(data), "Shapes must match");

        std::vector<Eigen::Triplet<T>> triplets;
        for (int row = 0; row < i.rows(); ++row) {
            triplets.emplace_back(Eigen::Triplet<T>(i(row), j(row), data(row)));
        }

        return triplets;
    }

    template<typename T>
    inline auto HStack(const std::vector<MatrixX<T>> &matrices) -> MatrixX<T> {
        const unsigned int cols = matrices.at(0).cols();
        const unsigned int total_rows = matrices.size() * matrices.at(0).rows();

        MatrixX<T> stacked(total_rows, cols);
        unsigned int current_row = 0;
        for (auto i = 0u; i < matrices.size(); ++i) {
            const MatrixX<T> mat = matrices.at(i);
            stacked.middleRows(current_row, mat.rows()) = mat;

            current_row += mat.rows();
        }

        return stacked;
    }

    template<typename T>
    inline auto HStack(const std::vector<VectorX<T>> &vectors) -> MatrixX<T> {
        const unsigned int rows = vectors.size();
        const unsigned int cols = vectors.at(0).cols();

        std::vector<MatrixX<T>> stacked;
        for (const VectorX<T> &vector : vectors) {
            const MatrixX<T> v = vector.transpose();
            stacked.emplace_back(v);
        }

        return HStack(stacked);
    }

    template<typename T>
    inline auto Find(const VectorX<T> &in, T value) -> VectorX<int> {
        std::vector<int> _out;
        for (int r = 0; r < in.rows(); ++r) {
            if (in(r) == value) { _out.push_back(r); }
        }

        VectorX<int> out(_out.size());
        for (int i = 0; i < _out.size(); ++i) { out(i) = _out.at(i); }

        return out;
    }

    template<typename Derived>
    inline auto Slice(const Eigen::DenseBase<Derived> &in, const int start, const int end,
                      Eigen::PlainObjectBase<Derived> &out) -> void {
        NEON_ASSERT_ERROR(start < end, "YOU PROVIDED AN INVALID SLICE RANGE");
        NEON_ASSERT_ERROR(start != end, "START AND END ARE THE SAME");
        NEON_ASSERT_ERROR(start < in.rows(), "START VALUE TOO LARGE");
        NEON_ASSERT_ERROR(end <= in.rows(), "END VALUE TOO LARGE");

        out.resize((end - start) + 1, in.cols());
        for (int i = start, out_idx = 0; i <= end; ++i, ++out_idx) { out(out_idx) = in(i); }
    }


    template<typename DerivedIn, typename DerivedOut, typename DerivedIndices>
    inline auto Slice(const Eigen::DenseBase<DerivedIn> &in, const Eigen::DenseBase<DerivedIndices> &rows,
                      const Eigen::DenseBase<DerivedIndices> &cols, Eigen::PlainObjectBase<DerivedOut> &out) -> void {
        NEON_ASSERT_ERROR(rows.minCoeff() >= 0, "ROW INDEX IS LESS THAN 0");
        NEON_ASSERT_ERROR(rows.maxCoeff() <= in.rows(), "ROW INDEX IS BIGGER THAN MAX SIZE OF INPUT MATRIX");
        NEON_ASSERT_ERROR(cols.minCoeff() >= 0, "COLUMN INDEX IS LESS THAN 0");
        NEON_ASSERT_ERROR(cols.maxCoeff() <= in.cols(), "COLUMN INDEX IS BIGGER THAN MAX SIZE OF INPUT MATRIX");
        out.resize(rows.size(), cols.size());

        for (int row = 0; row < rows.size(); ++row) {
            for (int col = 0; col < cols.size(); ++col) { out(row, col) = in(rows(row), cols(col)); }
        }
    }

    template<typename T>
    inline auto STLVectorToEigenVector(const std::vector<T> &in) -> VectorX<T> {
        VectorX<T> v(in.size());
        for (int row = 0; row < in.size(); ++row) { v(row) = in.at(row); }
        return v;
    }

    template<typename T>
    inline auto EigenVectorToSTLVector(const VectorX<T> &in) -> std::vector<T> {
        std::vector<T> v;
        v.reserve(in.rows());
        for (int row = 0; row < in.rows(); ++row) { v.push_back(in(row)); }

        return v;
    }


    /// \brief Sorts an Eigen Vector
    template<typename T>
    auto Sort(VectorX<T> &out) -> void {
        auto v = EigenVectorToSTLVector(out);
        std::sort(v.begin(), v.end());

        out = STLVectorToEigenVector(v);
    }

    /// \brief Removes duplicates from an Eigen Vector
    template<typename T>
    auto Dedupe(VectorX<T> &out) -> void {
        Sort(out);
        auto v = EigenVectorToSTLVector(out);
        v.erase(std::unique(v.begin(), v.end()), v.end());
        out = STLVectorToEigenVector(v);
    }

    template<typename DerivedIn>
    void NNZ(const Eigen::PlainObjectBase<DerivedIn> &in, VectorXi &I, unsigned int &count, bool rowwise) {
        const auto rows = in.rows();
        const auto cols = in.cols();

        std::vector<int> _I;

        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (in(row, col) > 0) {
                    if (rowwise) {
                        _I.push_back(row);
                    } else {
                        _I.push_back(col);
                    }
                }
            }
        }

        igl::list_to_matrix(_I, I);
        count = I.rows();
    }

    template<typename T>
    auto ComputeAngle(const Vector3<T> &a, const Vector3<T> &b) -> Real {
        return std::acos(a.dot(b) / (a.norm() * b.norm()));
    }

    template<typename T>
    void ComputeTriangleAngles(const Vector3<T> &a, const Vector3<T> &b, const Vector3<T> &c, Real &A_hat, Real &B_hat,
                               Real &C_hat) {
        const Real a_dist = Distance(b, c);
        const Real b_dist = Distance(a, c);
        const Real c_dist = Distance(a, b);

        // Now, using the law of cosines, get the angle for each value.
        A_hat = std::acos((b_dist * b_dist + c_dist * c_dist - a_dist * a_dist) / (2 * (b_dist * c_dist)));
        B_hat = std::acos((a_dist * a_dist + c_dist * c_dist - b_dist * b_dist) / (2 * (a_dist * c_dist)));
        C_hat = std::acos((b_dist * b_dist + a_dist * a_dist - c_dist * c_dist) / (2 * (b_dist * a_dist)));
    }

    template<typename T>
    auto Distance(const Vector3<T> &a, const Vector3<T> &b) -> Real {
        const Real dx = b.x() - a.x();
        const Real dy = b.y() - a.y();
        const Real dz = b.z() - a.z();
        return std::sqrt(std::pow(dx, 2) + std::pow(dy, 2) + std::pow(dz, 2));
    }

    template<typename Derived>
    auto Contains(const Eigen::MatrixBase<Derived> &in, const typename Derived::Scalar value) -> bool {
        const int rows = in.rows();
        const int cols = in.cols();

        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (in(row, col) == value) { return true; }
            }
        }

        return false;
    }

    inline auto IsApprox(const Real lhs, const Real rhs, const Real epsilon) -> bool {
        return std::abs(lhs - rhs) < epsilon;
    }

    inline auto Degrees(const Real degree) -> Real { return degree * (180.0 / kPi); }

    inline auto Radians(const Real radian) -> Real { return radian * (kPi / 180.0); }
}// namespace utilities::math


#endif//NEON_LINEARALGEBRA_H
