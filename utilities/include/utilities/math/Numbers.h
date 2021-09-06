// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_NUMBERS_H
#define NEON_NUMBERS_H

#include <utilities/math/LinearAlgebra.h>

namespace utilities::numbers {
    auto IsApprox(Real lhs, Real rhs, Real epsilon) -> bool;
}

#endif//NEON_NUMBERS_H
