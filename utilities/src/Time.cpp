// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#include <ctime>
#include <utilities/math/Time.h>

auto utilities::math::GetTimestamp() -> std::string {
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = std::localtime(&rawtime);

    strftime(buffer, sizeof(buffer), "%d-%m-%Y_%H:%M:%S", timeinfo);
    std::string str(buffer);
    return str;
}
