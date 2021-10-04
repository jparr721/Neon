// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Strivison <jarred.parr@ucdenver.edu>. All rights reserved.
//

#ifndef NEON_LINEAR_ELASTIC_50BY50_CUBE_H
#define NEON_LINEAR_ELASTIC_50BY50_CUBE_H

#include <utilities/math/LinearAlgebra.h>

namespace simulations::static_files {
    // Uniform Dimensions
    constexpr unsigned int kShape = 50;

    // Pounds
    constexpr int kForceMin = -1;
    constexpr int kForceMax = -100;

    // Youngs Modulus
    constexpr int kMinE = 30000;
    constexpr int kMaxE = 150000;
    constexpr int kEIncr = 1000;

    // Poisson's Ratio
    constexpr Real kMinv = 0.0;
    constexpr Real kMaxv = 0.5;
    constexpr Real kvIncr = 0.01;

    // Shear Modulus
    constexpr Real kMinG = 1e-5;// Rubber
    constexpr Real kMaxG = 480; // Diamond
    constexpr Real kGIncr = 1;

    // Thickness
    constexpr Real kMinT = 0.1;
    constexpr Real kMaxT = 0.7;
    constexpr Real kTIncr = 0.01;

    // Amplitude
    constexpr Real kMinA = 0.1;
    constexpr Real kMaxA = 0.9;
    constexpr Real kAIncr = 0.01;

}// namespace simulations::static_files

#endif//NEON_LINEAR_ELASTIC_50BY50_CUBE_H
