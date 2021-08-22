// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <solvers/materials/Material.h>
solvers::materials::Material
solvers::materials::MaterialFromLameCoefficients(unsigned int number, const std::string &name, Real G, Real lambda) {
    Material m;
    m.number = number;
    m.name = name;
    m.G = G;
    m.lambda = lambda;
    m.E = (G * (3 * lambda + 2 * G)) / (lambda + G);
    m.v = lambda / (2 * (lambda + G));

    return m;
}
solvers::materials::Material solvers::materials::MaterialFromEandv(unsigned int number, const std::string &name, Real E,
                                                                   Real v) {
    Material m;
    m.number = number;
    m.name = name;
    m.E = E;
    m.v = v;
    m.G = E / (2 * (1 + v));
    m.lambda = ((E * v) / (1 + v) * (1 - 2 * v));

    return m;
}
