// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_ALGORITHMS_H
#define NEON_ALGORITHMS_H

#include <functional>

namespace solvers::algorithms {
    struct FnBinarySearchAbstractReturnType {
        virtual auto TooLarge() const -> bool = 0;
        virtual auto TooSmall() const -> bool = 0;
        virtual auto Ok() const -> bool = 0;
    };

    /// \brief Binary Searches using an objective function and compares it to the return type
    template<typename Fn, typename Haystack, typename Needle>
    inline auto FnBinarySearch(const Fn &fn, const Haystack &container, const int &l, const int &r,
                               const Needle &value) {
        const int mid = l + (r - l) / 2;

        const auto ret = fn(value, container[mid]);

        if (ret.Ok()) {
            return ret;
        } else if (ret.TooLarge()) {
            return FnBinarySearch(fn, container, l, mid - 1, value);
        } else if (ret.TooSmall()) {
            return FnBinarySearch(fn, container, mid + 1, r, value);
        }

        return ret;
    }
}// namespace utilities::algorithms

#endif//NEON_ALGORITHMS_H
