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
        static constexpr int kNoThickness = -1;
        struct Inclusion {
            int n_inclusions;
            int depth;
            int rows;
            int cols;
        };

        static auto MakeInclusion(const int n, const int depth, const int rows, const int cols) -> Inclusion {
            return Inclusion{n, depth, rows, cols};
        }

        static auto MakeInclusion(const int n, const int dim) -> Inclusion { return Inclusion{n, dim, dim, dim}; }

        enum class GeneratorInfo {
            kSuccess = 0,
            kFailure,
        };

        enum class Behavior {
            kIsotropic = 0,
            kAnisotropic,
        };

        enum class Microstructure {
            kUniform = 0,
            kComposite,
        };

        ImplicitSurfaceGenerator(const unsigned int height, const unsigned int width, const unsigned int depth)
            : microstructure_(Microstructure::kUniform), behavior_(Behavior::kIsotropic) {
            implicit_surface_.Resize(height, width, depth);
            implicit_surface_.SetConstant(static_cast<T>(material_number_));
        }

        ImplicitSurfaceGenerator(const unsigned int height, const unsigned int width, const unsigned int depth,
                                 Behavior behavior, const Inclusion inclusion, const int &material_number = 1)
            : behavior_(behavior), microstructure_(Microstructure::kComposite), inclusion_(inclusion),
              material_number_(material_number) {
            implicit_surface_.Resize(height, width, depth);
            implicit_surface_.SetConstant(static_cast<T>(material_number_));
        }

        auto Surface() const -> Tensor3<T> { return implicit_surface_; }

        auto AddSquarePaddingLayers() -> Tensor3<T> {
            const int rows = implicit_surface_.Dimension(0) + 2;
            const int cols = implicit_surface_.Dimension(1) + 2;
            const int layers = implicit_surface_.Dimension(2) + 2;

            Tensor3<T> new_surface(rows, cols, layers);
            new_surface.SetConstant(0);

            for (int layer = 0; layer < layers; ++layer) {
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) {
                        if (layer == 0 || layer == layers - 1 || row == 0 || row == rows - 1 || col == 0 ||
                            col == cols - 1) {
                            continue;
                        } else {
                            const int r = row > 1 ? row - 1 : row;
                            const int l = layer > 1 ? layer - 1 : layer;
                            const int c = col > 1 ? col - 1 : col;
                            new_surface(row, col, layer) = implicit_surface_(r, c, l);
                        }
                    }
                }
            }

            return new_surface;
        }

        /// \brief Generate implicit function material.
        auto GenerateImplicitFunctionBasedMaterial(const int thickness, MatrixXr &V, MatrixXi &F) -> void {
            const int rows = implicit_surface_.Dimension(0);
            const int cols = implicit_surface_.Dimension(1);
            const int layers = implicit_surface_.Dimension(2);

            if (behavior_ == Behavior::kIsotropic) {
                if (microstructure_ == Microstructure::kUniform) {
                    MakeRenderable(V, F);
                } else {
                    GenerateIsotropicMaterial(thickness, V, F);
                }
            }
            if (behavior_ == Behavior::kAnisotropic) { GenerateAnisotropicMaterial(thickness, 1000, V, F); }
        }


    private:
        Behavior behavior_;
        Microstructure microstructure_;

        Tensor3<T> implicit_surface_;

        Inclusion inclusion_;

        int material_number_ = 1;

        std::vector<Cube> cubes_;

        GeneratorInfo info_ = GeneratorInfo::kSuccess;

        auto GenerateIsotropicMaterial(const int thickness, MatrixXr &V, MatrixXi &F) -> void {
            const int rows = implicit_surface_.Dimension(0);
            const int cols = implicit_surface_.Dimension(1);
            const int layers = implicit_surface_.Dimension(2);

            const int n_rows = inclusion_.n_inclusions / inclusion_.rows;
            const int n_cols = inclusion_.n_inclusions / n_rows;

            const int x_interval_pad = (cols - (inclusion_.cols + thickness) * n_cols) / 2;
            const int y_interval_pad = (rows - (inclusion_.rows + thickness) * n_rows) / 2;

            // Pre-calculate the starting zones for x
            VectorXi x_starting_positions =
                    VectorXi::LinSpaced(n_cols, x_interval_pad, cols - x_interval_pad - inclusion_.cols);

            if (cols - x_starting_positions(x_starting_positions.rows() - 1) != x_starting_positions.x()) {
                NEON_LOG_WARN("Uneven surface found! No longer isotropic. Attempting fix");
                FixStartingPositions(cols, x_starting_positions);
            }

            VectorXi y_starting_positions =
                    VectorXi::LinSpaced(n_rows, y_interval_pad, rows - y_interval_pad - inclusion_.rows);

            if (rows - y_starting_positions(y_starting_positions.rows() - 1) != y_starting_positions.x()) {
                NEON_LOG_WARN("Uneven y-axis surface found! Attempting to fix");
                FixStartingPositions(rows, y_starting_positions);
            }

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
            implicit_surface_.SetTopsBitmask(l.transpose());

            MakeRenderable(V, F);
        }

        auto GenerateAnisotropicMaterial(const unsigned int thickness, const unsigned int max_iter, MatrixXr &V,
                                         MatrixXi &F) -> void {
            constexpr unsigned int min = 0;

            // Ensure the cutoff conditions are met
            const unsigned int max_rows = implicit_surface_.Dimension(0) - (inclusion_.rows + thickness);
            const unsigned int max_cols = implicit_surface_.Dimension(1) - (inclusion_.cols + thickness);
            const unsigned int max_layers = implicit_surface_.Dimension(2) - (inclusion_.depth + thickness);

            // Iteration cutoff point for normalizing cubes
            int n_iterations = 0;

            // RNG
            std::random_device rd;
            std::mt19937 generator(rd());
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
                                return;
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

                    implicit_surface_(row, col, layer) = 0;
                }
            }

            MakeRenderable(V, F);
        }

        auto MakeRenderable(MatrixXr &V, MatrixXi &F) -> void {
            const Tensor3r mc_surface = AddSquarePaddingLayers();
            const int rows = mc_surface.Dimension(0);
            const int cols = mc_surface.Dimension(1);
            const int layers = mc_surface.Dimension(2);


            MatrixXr GV;
            RowVector3i resolution(rows, cols, layers);
            igl::grid(resolution, GV);

            const VectorXr Gf = mc_surface.Vector();

            try {
                igl::copyleft::marching_cubes(Gf, GV, rows, cols, layers, 0, V, F);
            } catch (...) { NEON_LOG_ERROR("Marching cubes failed"); }
        }


        auto FixStartingPositions(const int cols, VectorXi &x_starting_positions) -> void {
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
    };
}// namespace meshing


#endif//NEON_IMPLICITSURFACEGENERATOR_H
