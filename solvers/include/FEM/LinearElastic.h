//
// Created by jparr on 8/21/2021.
//

#ifndef NEON_LINEARELASTIC_H
#define NEON_LINEARELASTIC_H

#include <Eigen/Dense>

namespace neon::solvers {
    class LinearElastic {
    public:
        Eigen::MatrixXi foobar;
        auto Solve() -> void;
    };

}// namespace neon::solvers
#endif//NEON_LINEARELASTIC_H
