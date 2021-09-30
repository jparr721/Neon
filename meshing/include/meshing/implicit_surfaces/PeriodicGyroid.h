// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_PERIODICGYROID_H
#define NEON_PERIODICGYROID_H

#include <functional>
#include <utilities/math/LinearAlgebra.h>
#include <utilities/math/Tensors.h>

namespace meshing::implicit_surfaces {
    constexpr unsigned int kDefaultIso = 0;

    using GyroidImplicitFunction = std::function<Real(Real amplitude, const RowVector3r &)>;
    using dGyroidImplicitFunction = std::function<RowVector3r(GyroidImplicitFunction, const RowVector3r &)>;

    inline auto SineFunction(Real amplitude, const RowVector3r &pos) -> Real {
        const Real two_pi = (2.0 * utilities::math::kPi) / amplitude;
        const Real x = pos.x();
        const Real y = pos.y();
        const Real z = pos.z();
        return std::sin(two_pi * x) * std::cos(two_pi * y) + std::sin(two_pi * y) * std::cos(two_pi * z) +
               std::sin(two_pi * z) * std::cos(two_pi * x);
    };

    void ComputeImplicitGyroidDualContouring(Real amplitude, Real thickness, unsigned int resolution,
                                             const GyroidImplicitFunction &fn, const dGyroidImplicitFunction &dfn,
                                             MatrixXr &V, MatrixXi &F, Tensor3r &scalar_field);

    void ComputeImplicitGyroidMarchingCubes(Real amplitude, Real thickness, unsigned int resolution,
                                            const GyroidImplicitFunction &fn, MatrixXr &V, MatrixXi &F,
                                            Tensor3r &scalar_field);
}// namespace meshing::implicit_surfaces


#endif//NEON_PERIODICGYROID_H
