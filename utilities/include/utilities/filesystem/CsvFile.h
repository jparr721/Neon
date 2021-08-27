// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_CSVFILE_H
#define NEON_CSVFILE_H

#include <cassert>
#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <string>
#include <utilities/runtime/NeonLog.h>
#include <vector>

namespace utilities::filesystem {
    template<typename T>
    class CsvFile {
    public:
        CsvFile(const std::string &filename, const std::vector<std::string> &keys) : n_keys_(keys.size()) {
            fs_.open(filename, std::fstream::in | std::fstream::out | std::fstream::app);

            for (int i = 0; i < keys.size(); ++i) {
                const auto &key = keys.at(i);
                if (i < keys.size() - 1) {
                    fs_ << key << ",";
                } else {
                    fs_ << key;
                }
            }

            fs_ << std::endl;
        }

        ~CsvFile() {
            fs_.flush();
            fs_.close();
        }

        CsvFile &operator<<(const std::vector<T> &values) {
            assert(values.size() == n_keys_);
            for (int i = 0; i < values.size(); ++i) {
                const auto value = values.at(i);
                if (i < values.size() - 1) {
                    fs_ << value << ",";
                } else {
                    fs_ << value;
                }
            }

            fs_ << std::endl;

            return *this;
        }

    private:
        unsigned int n_keys_ = 0;

        std::fstream fs_;
    };
}// namespace utilities::filesystem

#endif//NEON_CSVFILE_H
