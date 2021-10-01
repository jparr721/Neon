// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <utilities/runtime/Profiler.h>

auto utilities::runtime::profiler::operator<<(std::ostream &os, const utilities::runtime::profiler::Profiler &profiler)
        -> std::ostream & {
    os << "Elapsed: " << profiler.end - profiler.start << "s";
    return os;
}

void utilities::runtime::profiler::Profiler::operator()() {
    if (start > 0) {
        end = GetTime();
    } else {
        start = GetTime();
    }
}
auto utilities::runtime::profiler::Profiler::GetTime() -> double {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void utilities::runtime::profiler::Profiler::Clear() {
    start = 0;
    end = 0;
}
