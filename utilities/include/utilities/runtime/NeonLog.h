// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_NEONLOG_H
#define NEON_NEONLOG_H

enum class NeonLogLevel {
    kInfo = 0,
    kWarn,
    kError,
};

namespace utilities::runtime {
    template<typename... Context>
    auto NeonLog(const NeonLogLevel level, const std::string &function, const std::string &file, const int line,
                 Context... context) -> void {
        const auto formatter = [&](const std::string &level) -> std::ostringstream {
            std::ostringstream oss;
            oss << "[" << level << "]"
                << "[" << file << ":" << line << "]" << function << "(): ";
            return oss;
        };

        switch (level) {
            case NeonLogLevel::kInfo: {
                std::ostringstream ss = formatter("INFO");
                LogAll(ss, context...);
                std::cout << ss.str() << std::endl;
                break;
            }
            case NeonLogLevel::kWarn: {
                std::ostringstream ss = formatter("WARN");
                LogAll(ss, context...);
                std::cerr << ss.str() << std::endl;
                break;
            }
            case NeonLogLevel::kError: {
                std::ostringstream ss = formatter("ERROR");
                LogAll(ss, context...);
                std::cerr << ss.str() << std::endl;
                break;
            }
        }
    }

    inline auto NeonLogSeparator() -> void { std::cout << "========================" << std::endl; }
}// namespace utilities::runtime

#define NEON_LOG_INFO(...)                                                                                             \
    utilities::runtime::NeonLog(NeonLogLevel::kInfo, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)

#define NEON_LOG_WARN(...)                                                                                             \
    utilities::runtime::NeonLog(NeonLogLevel::kWarn, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)

#define NEON_LOG_ERROR(...)                                                                                            \
    utilities::runtime::NeonLog(NeonLogLevel::kError, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)

#define NEON_LOG_SEPARATOR() utilities::runtime::NeonLogSeparator();

#endif//NEON_NEONLOG_H
