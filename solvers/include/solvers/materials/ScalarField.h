// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_SCALARFIELD_H
#define NEON_SCALARFIELD_H

#include <map>
#include <utilities/math/LinearAlgebra.h>
#include <utilities/math/Tensors.h>

namespace solvers::materials::scalar_field {
    /// Computes the material ratios
    /// \tparam T The type of the input tensor
    /// \param material The material scalar field
    /// \param ratios The ratio of each material in a sorted map
    template<typename T>
    void ComputeMaterialRatio(const Tensor3<T> &material, std::map<T, Real> &ratios) {
        const unsigned int rows = material.Dimension(0);
        const unsigned int cols = material.Dimension(1);
        const unsigned int layers = material.Dimension(2);

        for (auto row = 0; row < rows; ++row) {
            for (auto col = 0; col < cols; ++col) {
                for (auto layer = 0; layer < layers; ++layer) {
                    const auto value = material(row, col, layer);
                    ++ratios[value];
                }
            }
        }

        unsigned int sum = 0;
        for (const auto &[_, v] : ratios) { sum += v; }

        for (auto &[k, v] : ratios) { v /= sum; }
    }
}// namespace solvers::materials::scalar_field

#endif//NEON_SCALARFIELD_H
