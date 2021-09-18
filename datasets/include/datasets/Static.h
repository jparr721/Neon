// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_STATIC_H
#define NEON_STATIC_H

#include <utilities/math/LinearAlgebra.h>

namespace datasets {
    void MakeUniaxialStaticSolverDataset2D(Real force_min, Real force_max, unsigned int dim);
}

#endif//NEON_STATIC_H
