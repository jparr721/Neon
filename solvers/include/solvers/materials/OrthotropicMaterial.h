// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_ORTHOTROPICMATERIAL_H
#define NEON_ORTHOTROPICMATERIAL_H

#include <utilities/math/LinearAlgebra.h>

namespace solvers::materials {
    struct OrthotropicMaterial {
        VectorXr coefficients;
        Real E_x = 0.0;
        Real E_y = 0.0;
        Real E_z = 0.0;
        Real G_yz = 0.0;
        Real G_zx = 0.0;
        Real G_xy = 0.0;
        Real v_yx = 0.0;
        Real v_zx = 0.0;
        Real v_zy = 0.0;
        Real v_xy = 0.0;
        Real v_xz = 0.0;
        Real v_yz = 0.0;

        OrthotropicMaterial() = default;
        explicit OrthotropicMaterial(const VectorXr &coefficients);
        OrthotropicMaterial(Real E, Real v);

        auto ConstitutiveMatrix() -> Matrix6r;
    };
}// namespace solvers::materials

#endif//NEON_ORTHOTROPICMATERIAL_H
