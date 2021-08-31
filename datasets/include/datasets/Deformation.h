// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_DEFORMATION_H
#define NEON_DEFORMATION_H

#include <datasets/Dataset.h>
#include <utilities/math/LinearAlgebra.h>

namespace datasets {
    struct RowType {
        Real force;
        Real E;
        Real v;
        Real start;
        Real displacement;

        using ValueType = Real;

        auto Value(const std::string &key) const -> ValueType {
            if (key == "Force") {
                return force;
            } else if (key == "E") {
                return E;
            } else if (key == "v") {
                return v;
            } else if (key == "Start") {
                return start;
            } else if (key == "Displacement") {
                return displacement;
            }
            return 0;
        }
    };
    class Deformation : private Dataset<RowType> {
    public:
        explicit Deformation(const std::string &path) : Dataset(path) {}
        auto Read() -> void override;
        auto Value(const std::string &key) -> RowType::ValueType override { return rows_.at(0).Value(key); }

    private:
    };
}// namespace datasets

#endif//NEON_DEFORMATION_H
