//
// Created by jparr on 8/21/2021.
//

#ifndef NEON_NEONASSERT_H
#define NEON_NEONASSERT_H

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

enum class NeonAssertionLevel {
    kWarn = 0,
    kError,
};

namespace {
    template<typename T>
    void LogAll(std::ostringstream &o, T val) {
        o << val;
    }

    template<typename T, typename... Context>
    void LogAll(std::ostringstream &o, T val, Context... context) {
        LogAll(o, val);
        LogAll(o, context...);
    }
}// namespace

namespace utilities::assert {
    template<typename... Context>
    auto NeonAssert(const NeonAssertionLevel level, const bool condition, const std::string &function,
                    const std::string &file, const int line, Context... context) -> void {
        // Short circuit before we run through all the other logic.
        if (condition) { return; }

        const auto formatter = [](const std::string &func, const std::string &f, const int l) -> std::ostringstream {
            std::ostringstream ss;
            ss << "Assertion failed in " << f << "()[line:" << l << "] " << func << ": ";
            return ss;
        };
        switch (level) {
            case NeonAssertionLevel::kWarn: {
                std::ostringstream ss = formatter(function, file, line);
                LogAll(ss, context...);
                std::cerr << ss.str() << std::endl;
                break;
            }
            case NeonAssertionLevel::kError: {
                std::ostringstream ss = formatter(function, file, line);
                LogAll(ss, context...);
                throw std::runtime_error(ss.str());
            }
            default:
                throw std::runtime_error("CANNOT UNWIND ASSERTION");
        }
    }
}// namespace utilities::assert

#define NEON_ASSERT_WARN(cond, ...)                                                                                    \
    utilities::assert::NeonAssert(NeonAssertionLevel::kWarn, cond, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)

#define NEON_ASSERT_ERROR(cond, ...)                                                                                   \
    utilities::assert::NeonAssert(NeonAssertionLevel::kError, cond, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)

#endif//NEON_NEONASSERT_H
