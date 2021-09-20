// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_DYNAMIC_H
#define NEON_DYNAMIC_H

#include <utilities/math/LinearAlgebra.h>

namespace datasets {
    void MakeUniaxialDynamicSolverDataset(Real force, unsigned int dim, unsigned int entries);
}

#endif//NEON_DYNAMIC_H
