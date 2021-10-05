// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_PROFILER_H
#define NEON_PROFILER_H

#include <chrono>
#include <ostream>
#include <utilities/math/LinearAlgebra.h>

namespace solvers::runtime::profiler {
    class Profiler {
    public:
        double start = 0;
        double end = 0;

        void Clear();

        void operator()();
        friend auto operator<<(std::ostream &os, const Profiler &profiler) -> std::ostream &;

    private:
        auto GetTime() -> double;
    };
}// namespace utilities::runtime::profiler

#endif//NEON_PROFILER_H
