// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_HOMOGENIZATION_H
#define NEON_HOMOGENIZATION_H

#include <utilities/algorithms/Algorithms.h>
#include <utilities/math/LinearAlgebra.h>

namespace datasets {
    class Homogenization {
        struct HomogenizationBinarySearchReturnType : public utilities::algorithms::FnBinarySearchAbstractReturnType {
            VectorXr coefficients;
        };
    };
}// namespace datasets


#endif//NEON_HOMOGENIZATION_H
