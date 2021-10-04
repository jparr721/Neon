// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights
// reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public
// License v3. If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_IMPLICITSURFACEGENERATOR_H
#define NEON_IMPLICITSURFACEGENERATOR_H

#include <igl/copyleft/marching_cubes.h>
#include <igl/grid.h>
#include <igl/parallel_for.h>
#include <random>
#include <utilities/math/LinearAlgebra.h>
#include <utilities/math/Tensors.h>
#include <utilities/runtime/NeonAssert.h>
#include <utilities/runtime/NeonLog.h>
#include <vector>

namespace meshing::implicit_surfaces {
inline auto UnsignedVectorComparison(const Vector3<unsigned> &lhs,
                                     const Vector3<unsigned> &rhs) -> bool {
  if (lhs.x() < rhs.x())
    return true;
  if (rhs.x() < lhs.x())
    return false;
  if (lhs.y() < rhs.y())
    return true;
  if (rhs.y() < lhs.y())
    return false;
  return lhs.z() < rhs.z();
};

inline auto UnsignedVectorEqual(const Vector3<unsigned> &lhs,
                                const Vector3<unsigned> rhs) -> bool {
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
  Cube(int rows, int cols, int layers,
       const Vector3<unsigned int> &starting_index);

  auto Indices() const -> std::vector<Vector3<unsigned int>> {
    return indices_;
  }

  bool operator==(const Cube &rhs) const;

private:
  int rows_ = 0;
  int cols_ = 0;
  int layers_ = 0;
  std::vector<Vector3<unsigned int>> indices_;
};

template <typename T> class ImplicitSurfaceGenerator {
public:
  static constexpr int kNoThickness = -1;
  struct Inclusion {
    int n_inclusions;
    int depth;
    int rows;
    int cols;
  };

  static auto MakeInclusion(const int n, const int depth, const int rows,
                            const int cols) -> Inclusion {
    return Inclusion{n, depth, rows, cols};
  }

  static auto MakeInclusion(const int n, const int dim) -> Inclusion {
    return Inclusion{n, dim, dim, dim};
  }

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

  ImplicitSurfaceGenerator(const unsigned int height, const unsigned int width,
                           const unsigned int depth)
      : microstructure_(Microstructure::kUniform),
        behavior_(Behavior::kIsotropic) {
    implicit_surface_.Resize(height, width, depth);
    implicit_surface_.SetConstant(static_cast<T>(1));
  }

  ImplicitSurfaceGenerator(const unsigned int height, const unsigned int width,
                           const unsigned int depth, Behavior behavior,
                           Microstructure microstructure)
      : behavior_(behavior), microstructure_(microstructure) {
    implicit_surface_.Resize(height, width, depth);
    implicit_surface_.SetConstant(static_cast<T>(1));
  }

  ImplicitSurfaceGenerator(const unsigned int height, const unsigned int width,
                           const unsigned int depth, Behavior behavior,
                           const Inclusion inclusion,
                           const int &material_number = 1)
      : behavior_(behavior), microstructure_(Microstructure::kComposite),
        inclusion_(inclusion), material_number_(material_number) {
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

    for (int layer = 0; layer < layers - 2; ++layer) {
      for (int row = 0; row < rows - 2; ++row) {
        for (int col = 0; col < cols - 2; ++col) {
          const int r = row + 1;
          const int c = col + 1;
          const int l = layer + 1;
          new_surface(r, c, l) = implicit_surface_(row, col, layer);
        }
      }
    }

    return new_surface;
  }

  /// \brief Generate implicit function material.
  auto GenerateImplicitFunctionBasedMaterial(const int thickness,
                                             const bool auto_compute_area,
                                             MatrixXr &V, MatrixXi &F) -> void {
    if (behavior_ == Behavior::kIsotropic) {
      if (microstructure_ == Microstructure::kUniform) {
        MakeRenderable(V, F);
      } else {
        GenerateIsotropicMaterial(thickness, auto_compute_area, V, F);
      }
    }
    if (behavior_ == Behavior::kAnisotropic) {
      GenerateAnisotropicMaterial(thickness, 1000, V, F);
    }
  }

  auto Info() -> GeneratorInfo { return info_; }

private:
  Behavior behavior_;
  Microstructure microstructure_;

  Tensor3<T> implicit_surface_;

  Inclusion inclusion_;

  int material_number_ = 1;

  std::vector<Cube> cubes_;

  GeneratorInfo info_ = GeneratorInfo::kSuccess;

  auto GenerateIsotropicMaterial(const int thickness,
                                 const bool auto_compute_area, MatrixXr &V,
                                 MatrixXi &F) -> void {
    if (inclusion_.rows > 0) {
      NEON_ASSERT_ERROR(thickness > 0, "Thickness cannot be set to zero, "
                                       "otherwise we get really weird shit");
    }
    const unsigned int dim = implicit_surface_.Dimension(0);
    MatrixXr mask = MatrixXr::Ones(dim, dim);

    const unsigned int t_dim = dim - (2 * thickness);
    unsigned int v_dim = 0;
    if (auto_compute_area) {
      for (int i = 2; i < t_dim; ++i) {
        if (t_dim % i == 0) {
          v_dim = i;
          break;
        }
      }
      NEON_ASSERT_WARN(
          v_dim > 0,
          "Failed to compute optimum area. Defaulting to solid mesh");
      implicit_surface_.SetConstant(1);
      MakeRenderable(V, F);
    } else {
      v_dim = inclusion_.rows;
    }

    int material_rows = 0;
    for (int row = thickness; row < t_dim + thickness; ++row) {
      int material_cols = 0;
      int col = thickness;
      while (col < t_dim + thickness) {
        // We need to increment the rows so we don't draw voids in thickness
        // regions
        if (material_rows == v_dim) {
          // Add the spacing of size thickness
          row += thickness;

          // Reset the amount of material since we're back in a draw-able range
          material_rows = 0;
        }

        // We need to increment the cols so we don't draw voids in thickness
        // regions
        if (material_cols == v_dim) {
          // Add the spacing of size thickness
          col += thickness;

          // Reset the amount of material since we're back in a draw-able range
          material_cols = 0;
        } else { // Otherwise, increment as usual and draw the void.
          mask(row, col) = 0;

          // Since we drew material, increment material counts.
          ++material_cols;

          // Move to the next col since this one is drawn.
          ++col;
        }
      }

      // Material was drawn for this row, so we increment only here.
      ++material_rows;
    }

    for (int t = 0; t < thickness; ++t) {
      mask.row(t) = VectorXr::Ones(mask.rows());
      mask.col(t) = VectorXr::Ones(mask.cols());

      mask.row((mask.rows() - 1) - t) = VectorXr::Ones(mask.rows());
      mask.col((mask.cols() - 1) - t) = VectorXr::Ones(mask.cols());
    }

    // TODO(@jparr721) - Add checks for uniform_mesh even-ness

    // Set implicit surface to the bitmask
    implicit_surface_.SetConstant(1);
    implicit_surface_.SetLayersBitmask(mask);
    implicit_surface_.SetSidesBitmask(mask);
    implicit_surface_.SetTopsBitmask(mask);
    MakeRenderable(V, F);
  }

  auto GenerateAnisotropicMaterial(const unsigned int thickness,
                                   const unsigned int max_iter, MatrixXr &V,
                                   MatrixXi &F) -> void {
    // Ensure the cutoff conditions are met
    const unsigned int max_rows =
        implicit_surface_.Dimension(0) - (inclusion_.rows + thickness);
    const unsigned int max_cols =
        implicit_surface_.Dimension(1) - (inclusion_.cols + thickness);
    const unsigned int max_layers =
        implicit_surface_.Dimension(2) - inclusion_.depth;

    if (max_rows < inclusion_.n_inclusions &&
        max_cols < inclusion_.n_inclusions &&
        max_layers < inclusion_.n_inclusions) {
      NEON_ASSERT_ERROR("Provided anisotropic conditions cannot be met, the "
                        "available cutoff with supplied ",
                        "thickness is over-constrained");
      return;
    }

    // Iteration cutoff point for normalizing cubes
    int n_iterations = 0;

    // RNG
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<> rows_distribution(thickness, max_rows);
    std::uniform_int_distribution<> cols_distribution(thickness, max_cols);
    std::uniform_int_distribution<> layers_distribution(0, max_layers);

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

              c = Cube(inclusion_.rows, inclusion_.cols, inclusion_.depth,
                       start);
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
    const Tensor3<T> mc_surface = AddSquarePaddingLayers();
    const int rows = mc_surface.Dimension(0);
    const int cols = mc_surface.Dimension(1);
    const int layers = mc_surface.Dimension(2);

    MatrixXr GV;
    RowVector3i resolution(rows, cols, layers);
    igl::grid(resolution, GV);

    const VectorXr Gf = mc_surface.Vector();

    try {
      igl::copyleft::marching_cubes(Gf, GV, rows, cols, layers, 0, V, F);
    } catch (...) {
      NEON_LOG_ERROR("Marching cubes failed");
    }
  }

  auto FixStartingPositions(const int cols, VectorXi &x_starting_positions)
      -> void {
    const int start_padding = x_starting_positions.x();
    const int end_padding =
        cols - (x_starting_positions(x_starting_positions.rows() - 1) +
                inclusion_.cols);
    const int padding_diff = end_padding - start_padding;

    if (padding_diff % 2 == 0) {
      x_starting_positions.array() += padding_diff / 2;
      NEON_LOG_INFO("Successfully fixed improper padding.");
    } else {
      NEON_LOG_ERROR("Unable to fix isotropic padding due to invalid number of "
                     "available columns, have: ",
                     cols, " padding: ", padding_diff);
      return;
    }
  }
};
} // namespace meshing::implicit_surfaces

#endif // NEON_IMPLICITSURFACEGENERATOR_H
