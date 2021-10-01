// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//


#ifndef NEON_PIPELINE_H
#define NEON_PIPELINE_H

#include <string>
#include <utility>
#include <vector>

namespace pipelines {
    class Pipeline {
    public:
        explicit Pipeline(std::vector<std::string>  paths) : file_paths_(std::move(paths)) {}
        virtual ~Pipeline() = default;
        virtual void Run() = 0;

    protected:
        const std::vector<std::string> file_paths_;
    };
}

#endif//NEON_PIPELINE_H
