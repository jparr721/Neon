//
// Created by jparr on 8/21/2021.
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>

#ifndef NEON_LINEARALGEBRA_H
#define NEON_LINEARALGEBRA_H

#ifdef NEON_USE_DOUBLE
using Real = double;
#else
using Real = float;
#endif

// Dense Vector Types
using Vector2r = Eigen::Matrix<Real, 2, 1>;
using Vector3r = Eigen::Matrix<Real, 3, 1>;
using Vector4r = Eigen::Matrix<Real, 4, 1>;
using Vector6r = Eigen::Matrix<Real, 6, 1>;
using Vector12r = Eigen::Matrix<Real, 12, 1>;
using VectorXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

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

template<typename T>
using TensorReduction0 = Eigen::Tensor<T, 0>;

namespace utilities::linear_algebra {
    template<typename T>
    class Tensor3 {
    public:
        enum class OpOrientation {
            kRow = 0,
            kCol,
        };

        enum InsertOpIndex {
            kStart = 0,
            kEnd = -1,
        };

        Tensor3() = default;

        Tensor3(const Eigen::Tensor<T, 3> &instance) : instance_(instance) {}

        explicit Tensor3(const Vector3<int> &dims) { Resize(dims.x(), dims.y(), dims.z()); }

        Tensor3(int rows, int cols, int layers) { Resize(rows, cols, layers); }

        auto ToString() const -> std::string {
            std::stringstream ss;
            ss << instance_;
            return ss.str();
        }

        auto Matrix() const -> MatrixX<T> {
            const int rows = Dimension(0) * Dimension(2);
            const int cols = Dimension(1);
            return Matrix();
        }

        auto Vector() const -> VectorX<T> {
            const int rows = Dimensions().prod();
            const T *d = instance_.data();
            return Eigen::Map<const VectorX<T>>(d, rows);
        }

        auto Dimension(const int dim) const -> int { return instance_.dimension(dim); }

        auto Dimensions() const -> Vector3<int> { return Vector3<int>(Dimension(0), Dimension(1), Dimension(2)); }

        auto Instance() noexcept -> Eigen::Tensor<T, 3> & { return instance_; }
        auto Instance() const noexcept -> Eigen::Tensor<T, 3> { return instance_; }

        auto SetConstant(T value) -> void { instance_.setConstant(value); }

        auto Resize(int rows, int cols, int layers) -> void { instance_.resize(rows, cols, layers); }

        auto Sum() -> T {
            const TensorReduction0<T> sum = instance_.sum();
            return sum(0);
        }

        friend std::ostream &operator<<(std::ostream &out, const Tensor3 &t) { return out << t.instance_; }

        T &operator()(int row, int col, int layer) { return instance_(row, col, layer); }

        T operator()(int row, int col, int layer) const { return instance_(row, col, layer); }

        auto Layer(const int layer) const -> MatrixX<T> {
            MatrixX<T> m;
            const int rows = Dimension(0);
            const int cols = Dimension(1);

            m.resize(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) { m(i, j) = instance_(i, j, layer); }
            }

            return m;
        }

        auto SetLayer(const int idx, const MatrixX<T> &layer) -> void {
            const int rows = Dimension(0);
            const int cols = Dimension(1);

            numerics_assertion.Assert(layer.rows() == rows, __FUNCTION__, __FILE__, __LINE__,
                                      "Layer rows must match tensor dimensions");
            numerics_assertion.Assert(layer.cols() == cols, __FUNCTION__, __FILE__, __LINE__,
                                      "Layer cols must match tensor dimensions");

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) { instance_(row, col, idx) = layer(row, col); }
            }
        }

        auto Row(const int layer, const int row) const -> VectorX<T> {
            const int cols = Dimension(1);
            VectorX<T> v(cols);

            for (int col = 0; col < cols; ++col) { v(col) = instance_(row, col, layer); }

            return v;
        }

        auto Col(const int layer, const int col) const -> VectorX<T> {
            const int rows = Dimension(0);
            VectorX<T> v(rows);

            for (int row = 0; row < rows; ++row) { v(row) = instance_(row, col, layer); }

            return v;
        }

        auto Where(T value) const -> Tensor3<T> {
            const int rows = Dimension(0);
            const int cols = Dimension(1);
            const int layers = Dimension(2);
            Tensor3<T> output(rows, cols, layers);
            output.SetConstant(1);

            for (int layer = 0; layer < layers; ++layer) {
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) {
                        if (instance_(row, col, layer) != value) { output(row, col, layer) = 0; }
                    }
                }
            }

            return output;
        }

        auto WhereIdx(T value) const -> VectorX<int> {
            Tensor3<T> output = Where(value);
            VectorX<T> r = output.Vector();
            return linear_algebra::Find(r, value);
        }

        auto Append(const std::vector<VectorX<T>> &seqs, int index, OpOrientation orientation) -> Tensor3<T> {
            int rows = Dimension(0);
            int cols = Dimension(1);
            int layers = Dimension(2);

            numerics_assertion.Assert(seqs.size() == layers, __FUNCTION__, __FILE__, __LINE__,
                                      "Sequences must be able to broadcast across "
                                      "all layers, you provided: ",
                                      seqs.size(), " sequences, we need: ", layers);

            // Collapsed list of modified vectors in column-preserved order
            std::vector<VectorX<T>> collapsed_indices;
            if (orientation == OpOrientation::kCol) {
                ++cols;
                if (index == InsertOpIndex::kEnd) { index = cols - 1; }

                for (auto layer_idx = 0u; layer_idx < layers; ++layer_idx) {
                    const VectorX<T> new_col = seqs.at(layer_idx);
                    MatrixX<T> layer = Layer(layer_idx);
                    layer.conservativeResize(Eigen::NoChange, cols);

                    // If we aren't at the end, we need to "scoot" the other columns
                    // over.
                    if (index != cols - 1) {
                        for (int c = cols - 1; c > index; --c) { layer.col(c) = layer.col(c - 1); }
                        layer.col(index) = new_col;
                    } else {
                        // Otherwise, just drop it at the end.
                        layer.col(index) = new_col;
                    }

                    collapsed_indices.emplace_back(linear_algebra::MatrixToVector(layer));
                }
            }

            if (orientation == OpOrientation::kRow) {
                ++rows;
                if (index == InsertOpIndex::kEnd) { index = rows - 1; }

                for (auto layer_idx = 0u; layer_idx < layers; ++layer_idx) {
                    const VectorX<T> new_row = seqs.at(layer_idx);
                    MatrixX<T> layer = Layer(layer_idx);
                    layer.conservativeResize(rows, Eigen::NoChange);

                    // If we aren't at the end, we need to "scoot" the other rowumns
                    // over.
                    if (index != rows - 1) {
                        for (int r = rows - 1; r > index; --r) { layer.row(r) = layer.row(r - 1); }
                        layer.row(index) = new_row;
                    } else {
                        // Otherwise, just drop it at the end.
                        layer.row(index) = new_row;
                    }

                    collapsed_indices.emplace_back(linear_algebra::MatrixToVector(layer));
                }
            }

            // We need to re-flatten to preserve ordering. This is becuase
            // eigen tensors are column-major leading to misplaced indices
            // when rebuilding the index.
            VectorX<T> _d(collapsed_indices.at(0).rows() * layers, 1);
            int segment = 0;
            for (const VectorX<T> &v : collapsed_indices) {
                _d.segment(segment, v.rows()) = v;
                segment += v.rows();
            }

            return Tensor3<T>::Expand(_d, rows, cols, layers);
        }

        auto Append(const MatrixX<T> &layer, int index) -> Tensor3<T> {
            int rows = Dimension(0);
            int cols = Dimension(1);

            // Prep to add the next layer
            int layers = Dimension(2) + 1;

            numerics_assertion.Assert(layer.rows() == rows && layer.cols() == cols, __FUNCTION__, __FILE__, __LINE__,
                                      "Layer does not match dimensions, got: ", layer.rows(), " ", layer.cols(),
                                      " wanted: ", rows, " ", cols);

            // "resize" sweeps the tensor instance so, instead, make a new one.
            Tensor3<T> new_tensor(rows, cols, layers);

            if (index == InsertOpIndex::kEnd) {
                // Add the old indices back into the tensor
                for (int l = 0; l < layers - 1; ++l) {
                    for (int row = 0; row < rows; ++row) {
                        for (int col = 0; col < cols; ++col) { new_tensor(row, col, l) = instance_(row, col, l); }
                    }
                }

                // Insert the new last layer.
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) { new_tensor(row, col, layers - 1) = layer(row, col); }
                }
            }

            if (index == InsertOpIndex::kStart) {
                // Insert the new first layer.
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) { new_tensor(row, col, 0) = layer(row, col); }
                }

                // Add the old indices back into the tensor
                for (int l = 0; l < layers - 1; ++l) {
                    for (int row = 0; row < rows; ++row) {
                        for (int col = 0; col < cols; ++col) { new_tensor(row, col, l + 1) = instance_(row, col, l); }
                    }
                }
            }
            return new_tensor;
        }

        template<typename... Indices>
        auto At(Indices &&...indices) const {
            constexpr std::size_t n_indices = sizeof...(indices);

            if constexpr (n_indices == 1) { return Layer(indices...); }

            if constexpr (n_indices == 2) { return Row(indices...); }

            if constexpr (n_indices == 3) { return instance_(indices...); }
        }

        template<typename Derived>
        static auto Expand(const Eigen::PlainObjectBase<Derived> &in, int x_dim, int y_dim, int z_dim) -> Tensor3<T> {
            return Tensor3<T>(Eigen::TensorMap<Eigen::Tensor<const T, 3>>(in.data(), x_dim, y_dim, z_dim));
        }

        static auto FromStack(const std::vector<MatrixX<T>> &stack) {
            Tensor3<T> out(stack.at(0).rows(), stack.at(0).cols(), stack.size());
            for (int i = 0; i < stack.size(); ++i) { out.SetLayer(i, stack.at(i)); }

            return out;
        }

        static auto Replicate(const MatrixX<T> &layer, int times) -> Tensor3<T> {
            Tensor3<T> out(layer.rows(), layer.cols(), times);
            for (int i = 0; i < times; ++i) { out.SetLayer(i, layer); }

            return out;
        }

    private:
        Eigen::Tensor<T, 3> instance_;
    };

    using Tensor3r = Tensor3<Real>;
    using Tensor3i = Tensor3<int>;
}// namespace utilities::linear_algebra

#endif//NEON_LINEARALGEBRA_H
