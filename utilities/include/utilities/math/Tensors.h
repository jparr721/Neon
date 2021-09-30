// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_TENSORS_H
#define NEON_TENSORS_H

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utilities/math/LinearAlgebra.h>
#include <utilities/runtime/NeonAssert.h>
#include <utilities/runtime/NeonLog.h>

template<typename T>
using TensorReduction0 = Eigen::Tensor<T, 0>;

template<typename T>
using Tensor4 = Eigen::Tensor<T, 4>;

template<typename T>
using Tensor5 = Eigen::Tensor<T, 5>;

namespace utilities::math {

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

        explicit Tensor3(const Eigen::Tensor<T, 3> &instance) : instance_(instance) {}
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
            const T *d = instance_.data();
            return Eigen::Map<const MatrixX<T>>(d, rows, cols);
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
        static auto SetConstant(T value, const Vector3<int> &dims) -> Tensor3<T> {
            Tensor3<T> v(dims);
            v.SetConstant(value);
            return v;
        }

        auto Resize(int rows, int cols, int layers) -> void { instance_.resize(rows, cols, layers); }

        auto Sum() -> T {
            const TensorReduction0<T> sum = instance_.sum();
            return sum(0);
        }

        friend std::ostream &operator<<(std::ostream &out, const Tensor3 &t) { return out << t.instance_; }

        auto operator()(int row, int col, int layer) -> T & { return instance_(row, col, layer); }

        auto operator()(int row, int col, int layer) const -> T { return instance_(row, col, layer); }

        auto MakeBinary(Real cutoff = 0.0) -> Tensor3<T> {
            Tensor3<T> out(instance_);

            const int rows = Dimension(0);
            const int cols = Dimension(1);
            const int layers = Dimension(2);

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    for (int layer = 0; layer < layers; ++layer) {
                        const auto value = out(row, col, layer);
                        if (value <= cutoff) {
                            out(row, col, layer) = 0;
                        } else {
                            out(row, col, layer) = 1;
                        }
                    }
                }
            }

            return out;
        }

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

        auto SetLayerBitmask(const int layer, const MatrixX<T> &data) -> void {
            const int rows = Dimension(0);
            const int cols = Dimension(1);

            NEON_ASSERT_ERROR(data.rows() == rows, "Layer rows must match tensor row dimensions");
            NEON_ASSERT_ERROR(data.cols() == cols, "Layer cols must match tensor column dimensions");

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    if (data(row, col) == 0) { instance_(row, col, layer) = data(row, col); }
                }
            }
        }

        auto SetLayersBitmask(const MatrixX<T> &data) -> void {
            const int layers = Dimension(2);
            for (int layer = 0; layer < layers; ++layer) { SetLayerBitmask(layer, data); }
        }

        auto Top(const int layer) const -> MatrixX<T> {
            const int rows = Dimension(1);
            const int cols = Dimension(2);

            MatrixX<T> data(rows, cols);

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) { data(row, col) = instance_(layer, row, col); }
            }

            return data;
        }

        auto SetTop(const int layer, const MatrixX<T> &data) -> void {
            const int rows = Dimension(1);
            const int cols = Dimension(2);

            NEON_ASSERT_ERROR(data.rows() == rows, "Layer rows must match tensor column dimensions");
            NEON_ASSERT_ERROR(data.cols() == cols, "Layer cols must match tensor layer dimensions");

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) { instance_(layer, row, col) = data(row, col); }
            }
        }

        auto SetTopBitmask(const int layer, const MatrixX<T> &data) -> void {
            const int rows = Dimension(1);
            const int cols = Dimension(2);

            NEON_ASSERT_ERROR(data.rows() == rows, "Layer rows must match tensor column dimensions");
            NEON_ASSERT_ERROR(data.cols() == cols, "Layer cols must match tensor layer dimensions");

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    if (data(row, col) == 0) { instance_(layer, row, col) = data(row, col); }
                }
            }
        }

        auto SetTops(const MatrixX<T> &data) -> void {
            const int layers = Dimension(1);
            for (int layer = 0; layer < layers; ++layer) { SetTop(layer, data); }
        }

        auto SetTopsBitmask(const MatrixX<T> &data) -> void {
            const int layers = Dimension(1);
            for (int layer = 0; layer < layers; ++layer) { SetTopBitmask(layer, data); }
        }

        auto Side(const int layer) const -> MatrixX<T> {
            const int rows = Dimension(0);
            const int cols = Dimension(2);
            MatrixX<T> data(rows, cols);

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) { data(row, col) = instance_(row, layer, col); }
            }

            return data;
        }

        auto SetSide(const int layer, const MatrixX<T> &data) -> void {
            const int rows = Dimension(0);
            const int cols = Dimension(2);

            NEON_ASSERT_ERROR(data.rows() == rows, "Layer rows must match tensor dimensions");
            NEON_ASSERT_ERROR(data.cols() == cols, "Layer cols must match tensor dimensions");

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) { instance_(row, layer, col) = data(row, col); }
            }
        }

        auto SetSideBitmask(const int layer, const MatrixX<T> &data) -> void {
            const int rows = Dimension(0);
            const int cols = Dimension(2);

            NEON_ASSERT_ERROR(data.rows() == rows, "Layer rows must match tensor dimensions");
            NEON_ASSERT_ERROR(data.cols() == cols, "Layer cols must match tensor dimensions");

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    if (data(row, col) == 0) { instance_(row, layer, col) = data(row, col); }
                }
            }
        }

        auto SetSides(const MatrixX<T> &data) -> void {
            const int layers = Dimension(1);

            for (int layer = 0; layer < layers; ++layer) { SetSide(layer, data); }
        }

        auto SetSidesBitmask(const MatrixX<T> &data) -> void {
            const int layers = Dimension(1);

            for (int layer = 0; layer < layers; ++layer) { SetSideBitmask(layer, data); }
        }

        auto SetLayer(const int layer, const MatrixX<T> &data) -> void {
            const int rows = Dimension(0);
            const int cols = Dimension(1);

            NEON_ASSERT_ERROR(data.rows() == rows, "Layer rows must match tensor dimensions");
            NEON_ASSERT_ERROR(data.cols() == cols, "Layer cols must match tensor dimensions");

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) { instance_(row, col, layer) = data(row, col); }
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

        auto SetCol(const int layer, const int col, const VectorX<T> &data) -> void {
            NEON_ASSERT_ERROR(data.rows() == Dimension(0),
                              "Rows of new data must match the existing dimensions of the tensor");
            const int rows = Dimension(0);

            for (int row = 0; row < rows; ++row) { instance_(row, col, layer) = data(row); }
        }

        auto SetColConstant(const int layer, const int col, const T value) -> void {
            const VectorX<T> data = VectorX<T>::Constant(Dimension(0), value);
            SetCol(layer, col, data);
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
            return math::Find(r, value);
        }

        auto Append(const std::vector<VectorX<T>> &seqs, int index, OpOrientation orientation) -> Tensor3<T> {
            int rows = Dimension(0);
            int cols = Dimension(1);
            int layers = Dimension(2);

            NEON_ASSERT_ERROR(seqs.size() == layers,
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

                    collapsed_indices.emplace_back(math::MatrixToVector(layer));
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

                    collapsed_indices.emplace_back(math::MatrixToVector(layer));
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

            NEON_ASSERT_ERROR(layer.rows() == rows && layer.cols() == cols,
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

        static auto FromStack(const std::vector<MatrixX<T>> &stack) -> Tensor3<T> {
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

    namespace tensors {
        template<typename T>
        void Assign(Tensor4<T> &tensor, const unsigned int dim2, const unsigned int dim3, MatrixX<T> &data) {
            const unsigned int rows = tensor.dimension(0);
            NEON_ASSERT_ERROR(rows == data.rows(), "Data rows and tensor dim 0 must match");
            const unsigned int cols = tensor.dimension(1);
            NEON_ASSERT_ERROR(cols == data.cols(), "Data cols and tensor dim 0 must match");

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) { tensor(row, col, dim2, dim3) = data(row, col); }
            }
        }

        template<typename T, int dim>
        void Write(const Eigen::Tensor<T, dim> &t, const std::string &filename) {
            NEON_ASSERT_ERROR(!std::filesystem::exists(filename), "File already exists");
            // Do not append
            std::fstream fs;
            fs.open(filename, std::fstream::in | std::fstream::out | std::fstream::app);

            std::ostringstream oss;
            for (int i = 0; i < t.dimensions().size(); ++i) { oss << t.dimension(i) << " "; }
            oss << std::endl;
            oss << t << std::endl;
            fs << oss.str();

            fs.flush();
            fs.close();
        }
    }// namespace tensors

}// namespace utilities::math

template<typename T>
using Tensor3 = utilities::math::Tensor3<T>;
using Tensor3r = utilities::math::Tensor3<Real>;
using Tensor3i = utilities::math::Tensor3<int>;

#endif//NEON_TENSORS_H
