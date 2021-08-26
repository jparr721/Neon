// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_IMPLICITSURFACEGENERATOR_H
#define NEON_IMPLICITSURFACEGENERATOR_H

#include <igl/copyleft/marching_cubes.h>
#include <igl/grid.h>
#include <igl/parallel_for.h>
#include <random>
#include <utilities/math/LinearAlgebra.h>
#include <utilities/runtime/NeonAssert.h>
#include <utilities/runtime/NeonLog.h>
#include <vector>

namespace meshing {
    inline auto UnsignedVectorComparison(const Vector3<unsigned> &lhs, const Vector3<unsigned> &rhs) -> bool {
        if (lhs.x() < rhs.x()) return true;
        if (rhs.x() < lhs.x()) return false;
        if (lhs.y() < rhs.y()) return true;
        if (rhs.y() < lhs.y()) return false;
        return lhs.z() < rhs.z();
    };

    inline auto UnsignedVectorEqual(const Vector3<unsigned> &lhs, const Vector3<unsigned> rhs) -> bool {
        return lhs.x() == rhs.x() && lhs.y() == rhs.y() && lhs.z() == rhs.z();
    };

    class Cube {
    public:
        /// \brief Cube is an anonymous cube structure for cuboid-based inclusions.
        /// @param rows The rows of the cube
        /// @param cols The cols of the cube
        /// @param layers The layers of the cube
        /// @param starting_index The xyz coordinate of the
        /// bottom-left position to build the cube from
        Cube(int rows, int cols, int layers, const Vector3<unsigned int> &starting_index);

        auto Indices() const -> std::vector<Vector3<unsigned int>> { return indices_; }

        bool operator==(const Cube &rhs) const;

    private:
        int rows_ = 0;
        int cols_ = 0;
        int layers_ = 0;
        std::vector<Vector3<unsigned int>> indices_;
    };

    template<typename T>
    class ImplicitSurfaceGenerator {
    public:
        struct Inclusion {
            int n_inclusions;
            int area;
            int depth;
            int rows;
            int cols;
        };

        enum class GeneratorInfo {
            kSuccess = 0x00,
            kFailure = 0x01,
        };

        enum class ImplicitSurfaceCharacteristics {
            kIsotropic = 0x00,
            kAnisotropic = 0x01,
        };

        enum class ImplicitSurfaceMicrostructure {
            kUniform = 0x00,
            kComposite = 0x01,
        };

        int minimum_surface_padding = 1;

        ImplicitSurfaceGenerator(const unsigned int height, const unsigned int width, const unsigned int depth,
                                 ImplicitSurfaceCharacteristics behavior, ImplicitSurfaceMicrostructure microstructure,
                                 const Inclusion inclusion, const int &material_1_number, const int &material_2_number)
            : behavior_(behavior), microstructure_(microstructure), inclusion_(inclusion),
              material_1_number_(material_1_number), material_2_number_(material_2_number) {
            implicit_surface_.Resize(height, width, depth);
            implicit_surface_.SetConstant(static_cast<T>(material_1_number_));
        }

        ImplicitSurfaceGenerator(const unsigned int height, const unsigned int width, const unsigned int depth,
                                 const int &material_1_number)
            : material_1_number_(material_1_number), microstructure_(ImplicitSurfaceMicrostructure::kUniform),
              behavior_(ImplicitSurfaceCharacteristics::kIsotropic) {
            implicit_surface_.Resize(height, width, depth);
            implicit_surface_.SetConstant(static_cast<T>(material_1_number_));
        }

        auto Surface() const -> Tensor3<T> { return implicit_surface_; }

        auto Generate() -> Tensor3<T> {
            if (microstructure_ == ImplicitSurfaceMicrostructure::kComposite) {
                behavior_ == ImplicitSurfaceCharacteristics::kIsotropic ? GenerateIsotropicMaterial()
                                                                        : GenerateAnisotropicMaterial();
            }

            return implicit_surface_;
        }

        auto AddSquarePaddingLayers() -> Tensor3<T> {
            const int rows = implicit_surface_.Dimension(0);
            const int cols = implicit_surface_.Dimension(1);
            const int layers = implicit_surface_.Dimension(2);

            Tensor3<T> new_surface(rows, cols, layers);

            for (int layer = 0; layer < layers; ++layer) {
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) {
                        if (layer == 0 || layer == layers - 1 || row == 0 || row == rows - 1 || col == 0 ||
                            col == cols - 1) {
                            new_surface(row, col, layer) = 0;
                        } else {
                            new_surface(row, col, layer) = implicit_surface_(row, col, layer);
                        }
                    }
                }
            }

            return new_surface;
        }

        auto GenerateImplicitFunctionBasedMaterial(MatrixXr &V, MatrixXi &F, const int thickness = 1) -> void {
            const int rows = implicit_surface_.Dimension(0);
            const int cols = implicit_surface_.Dimension(1);
            const int layers = implicit_surface_.Dimension(2);

            MatrixXr GV;
            RowVector3i resolution(rows, cols, layers);
            igl::grid(resolution, GV);

            if (behavior_ == ImplicitSurfaceCharacteristics::kIsotropic) {
                const int n_rows = inclusion_.n_inclusions / inclusion_.rows;
                const int n_cols = inclusion_.n_inclusions / n_rows;
                const int x_interval_pad = (rows - (inclusion_.cols + thickness) * n_cols) / 2;

                // Pre-calculate the starting zones for x
                VectorXi x_starting_positions =
                        VectorXi::LinSpaced(n_cols, x_interval_pad, cols - x_interval_pad - inclusion_.cols);

                if (cols - x_starting_positions(x_starting_positions.rows() - 1) != x_starting_positions.x()) {
                    NEON_LOG_WARN("Uneven surface found! No longer isotropic. Attempting fix");
                    FixColumnStartingPositions(cols, x_starting_positions);
                }
                const VectorXi y_starting_positions =
                        VectorXi::LinSpaced(n_rows, inclusion_.rows, rows - inclusion_.rows * 2);


                implicit_surface_.SetConstant(1);
                for (int layer = 1; layer < layers - 1; ++layer) {
                    for (int y_i = 0; y_i < n_rows; ++y_i) {
                        const int y = y_starting_positions(y_i);
                        for (int j = y; j < y + inclusion_.rows; ++j) {
                            for (int x_i = 0; x_i < n_cols; ++x_i) {
                                const int x = x_starting_positions(x_i);
                                for (int i = x; i < x + inclusion_.cols; ++i) { implicit_surface_(j, i, layer) = 0; }
                            }
                        }
                    }
                }

                const MatrixXr l = implicit_surface_.Layer(2);
                implicit_surface_.SetSidesBitmask(l);
                implicit_surface_.SetTopsBitmask(l);

                implicit_surface_ = AddSquarePaddingLayers();
                const VectorXr Gf = implicit_surface_.Vector();

                igl::copyleft::marching_cubes(Gf, GV, rows, cols, layers, 0, V, F);
            }
        }

        auto Info() -> GeneratorInfo { return info_; }

        auto FixColumnStartingPositions(const int cols, VectorXi &x_starting_positions) -> void {
            const int start_padding = x_starting_positions.x();
            const int end_padding = cols - (x_starting_positions(x_starting_positions.rows() - 1) + inclusion_.cols);
            const int padding_diff = end_padding - start_padding;

            if (padding_diff % 2 == 0) {
                x_starting_positions.array() += padding_diff / 2;
                NEON_LOG_INFO("Successfully fixed improper padding.");
            } else {
                NEON_LOG_ERROR("Unable to fix isotropic padding due to invalid number of available columns, have: ",
                               cols, " padding: ", padding_diff);
                return;
            }
        }

    private:
        ImplicitSurfaceCharacteristics behavior_;

        ImplicitSurfaceMicrostructure microstructure_;

        Tensor3<T> implicit_surface_;

        Inclusion inclusion_;

        int material_1_number_;
        int material_2_number_;

        std::vector<Cube> cubes_;

        GeneratorInfo info_ = GeneratorInfo::kSuccess;

        // Helpers ======================
        auto LayerContainsSecondaryMaterial(const MatrixXr &layer) -> bool {
            const auto rows = layer.rows();
            const auto cols = layer.cols();

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    if (layer(row, col) == material_2_number_) { return true; }
                }
            }

            return false;
        }

        /*
        @brief Ensures padding is even on all sides, returning false if not. Only
        applies to isotropic material meshes.
        */
        auto CheckLayerPadding(const VectorXr &layer, int max) -> void {
            const auto left_padding = layer(0) + 1;
            const auto right_padding = max - layer(layer.rows() - 1);

            NEON_ASSERT_WARN(left_padding == right_padding, "Padding does not match: ", left_padding, right_padding,
                             layer);
        }

        auto GenerateIsotropicMaterial() -> Tensor3<T> {
            // Y axis origin with padding
            VectorXr y_axis_origins = utilities::math::LinSpace(
                    inclusion_.area + minimum_surface_padding,
                    (implicit_surface_.Dimension(1) + 1 - (inclusion_.area + minimum_surface_padding)),
                    inclusion_.rows);
            y_axis_origins -= VectorXr::Ones(y_axis_origins.rows());
            CheckLayerPadding(y_axis_origins, implicit_surface_.Dimension(1));

            VectorXr x_axis_origins = utilities::math::LinSpace(
                    inclusion_.area + minimum_surface_padding,
                    (implicit_surface_.Dimension(0) + 1 - (inclusion_.area + minimum_surface_padding)),
                    inclusion_.cols);
            x_axis_origins -= VectorXr::Ones(x_axis_origins.rows());
            CheckLayerPadding(x_axis_origins, implicit_surface_.Dimension(0));

            std::vector<Vector2<unsigned int>> centroids;
            for (int i = 0; i < y_axis_origins.rows(); ++i) {
                for (int j = 0; j < x_axis_origins.rows(); ++j) {
                    const int y = static_cast<int>(y_axis_origins(i));
                    const int x = static_cast<int>(x_axis_origins(j));
                    centroids.emplace_back(x, y);
                }
            }

            const VectorXr layer_layout =
                    utilities::math::LinSpace(0, implicit_surface_.Dimension(2) - inclusion_.depth,
                                              implicit_surface_.Dimension(2) / inclusion_.depth);

            std::vector<Vector3<unsigned int>> indices;
            for (int i = 0; i < layer_layout.rows(); ++i) {
                const int layer = static_cast<int>(layer_layout(i));
                for (const auto &centroid : centroids) {
                    const auto cube_indices = MakeShapedIndices(
                            centroid, Vector3<unsigned int>(inclusion_.area, inclusion_.area, inclusion_.depth), layer);
                    indices.insert(indices.end(), cube_indices.begin(), cube_indices.end());
                }
            }

            SetFromIndices(indices);

            return implicit_surface_;
        }

        auto GenerateAnisotropicMaterial(const unsigned int size_variability = 1,
                                         const unsigned int min_inclusion_separation = 1,
                                         const unsigned int max_iter = 1000) -> Tensor3<T> {
            constexpr unsigned int min = 0;

            // Ensure the cutoff conditions are met
            const unsigned int max_rows = implicit_surface_.Dimension(0) - inclusion_.rows;
            const unsigned int max_cols = implicit_surface_.Dimension(1) - inclusion_.cols;
            const unsigned int max_layers = implicit_surface_.Dimension(2) - inclusion_.depth;

            // Iteration cutoff point for normalizing cubes
            int n_iterations = 0;

            // RNG
            std::default_random_engine generator;
            const std::uniform_int_distribution<int> rows_distribution(min, max_rows);
            const std::uniform_int_distribution<int> cols_distribution(min, max_cols);
            const std::uniform_int_distribution<int> layers_distribution(min, max_layers);

            for (unsigned int i = 0; i < inclusion_.n_inclusions; ++i) {
                unsigned int rows = rows_distribution(generator);
                unsigned int cols = cols_distribution(generator);
                unsigned int layers = layers_distribution(generator);
                Vector3<unsigned int> start(rows, cols, layers);

                auto c = Cube(inclusion_.rows, inclusion_.cols, inclusion_.depth, start);

                if (!cubes_.empty()) {
                    for (const Cube &cube : cubes_) {
                        if (cube == c) {
                            while (cube == c && n_iterations < max_iter) {
                                rows = rows_distribution(generator);
                                cols = cols_distribution(generator);
                                layers = layers_distribution(generator);
                                start = Vector3<unsigned int>(rows, cols, layers);

                                c = Cube(inclusion_.rows, inclusion_.cols, inclusion_.depth, start);
                                ++n_iterations;
                            }

                            if (n_iterations >= max_iter) {
                                info_ = GeneratorInfo::kFailure;
                                return implicit_surface_;
                            } else {
                                n_iterations = 0;
                            }
                        }
                    }
                }

                // TODO(@jparr721) - Tri-axial rotation
                cubes_.emplace_back(c);
            }

            for (const Cube &cube : cubes_) {
                for (const Vector3<unsigned int> &index : cube.Indices()) {
                    const unsigned int row = index.x();
                    const unsigned int col = index.y();
                    const unsigned int layer = index.z();

                    implicit_surface_(row, col, layer) = static_cast<T>(material_2_number_);
                }
            }

            return implicit_surface_;
        }


        // Shape Generators
        auto MakeShapedIndices(const Vector2<unsigned> &centroid, const Vector3<unsigned> &shape,
                               unsigned int layer_number) -> std::vector<Vector3<unsigned int>> {
            std::vector<Vector3<unsigned int>> indices;
            const unsigned int width{shape.x() / 2};
            const unsigned int height{shape.y() / 2};
            const unsigned int depth = shape.z();

            const unsigned int current_x = centroid.x();
            const unsigned int current_y = centroid.y();
            for (auto layer = layer_number; layer < layer_number + depth; ++layer) {
                for (auto x = 0u; x < width; ++x) {
                    for (auto y = 0u; y < height; ++y) {
                        indices.emplace_back(current_x + x, current_y + y, layer);
                        indices.emplace_back(current_x - x, current_y - y, layer);
                        indices.emplace_back(current_x - x, current_y + y, layer);
                        indices.emplace_back(current_x + x, current_y - y, layer);
                    }
                }
            }

            // TODO(@jparr721) - This is stupid and should be re-written.
            std::sort(indices.begin(), indices.end(), UnsignedVectorComparison);
            indices.erase(std::unique(indices.begin(), indices.end(), UnsignedVectorEqual), indices.end());

            return indices;
        }

        auto SetFromIndices(const std::vector<Vector3<unsigned int>> &indices) -> void {
            for (const Vector3<unsigned int> &index : indices) {
                implicit_surface_(index.x(), index.y(), index.z()) = material_2_number_;
            }
        }
    };
}// namespace meshing


#endif//NEON_IMPLICITSURFACEGENERATOR_H
