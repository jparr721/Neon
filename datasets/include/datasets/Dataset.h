// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_DATASET_H
#define NEON_DATASET_H

#include <string>
#include <utility>
#include <vector>

namespace datasets {
    template<typename DatasetRowType>
    class Dataset {
    public:
        explicit Dataset(std::string path) : path_(std::move(path)) {}
        virtual auto Read() -> void = 0;
        virtual auto Value(const std::string &key) -> typename DatasetRowType::ValueType = 0;

    protected:
        const std::string path_;
        std::vector<std::string> keys_;
        std::vector<DatasetRowType> rows_;
    };
}// namespace datasets

#endif//NEON_DATASET_H
