// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <string>
#include <utilities/math/LinearAlgebra.h>

#ifndef NEON_MATERIAL_H
#define NEON_MATERIAL_H

namespace solvers::materials {
    struct Material {
        // The number this material represents in the scalar field
        unsigned int number = 1;

        // Name of the material, for bookkeeping
        std::string name;

        // Young's Modulus
        Real E = -1;

        // Poisson's Ratio
        Real v = -1;

        // Shear Modulus
        Real G = -1;

        // Lame's Lambda
        Real lambda = -1;

        [[nodiscard]] auto IsInit() const noexcept -> bool { return !name.empty(); }
    };

    Material MaterialFromLameCoefficients(unsigned int number, const std::string &name, Real G, Real lambda);

    Material MaterialFromEandv(unsigned int number, const std::string &name, Real E, Real v);
}// namespace solvers::materials

#endif//NEON_MATERIAL_H
