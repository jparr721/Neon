#include <utilities/math/LinearAlgebra.h>

auto solvers::math::LinSpace(Real start, Real stop, unsigned int num) -> VectorXr {
    VectorXr interval;
    if (num == 1) {
        interval.resize(1);
        interval << (stop - start / 2) + start;
        return interval;
    }

    NEON_ASSERT_ERROR(stop > start, "STOP CANNOT BE LESS THAN START", "Stop: ", stop, "Start: ", start);

    const Real div = num - 1;

    NEON_ASSERT_ERROR(div > 0, "NUM MUST BE GREATER THAN 1");

    const Real delta = stop - start;
    interval.resize(num);

    // Initialize
    for (int i = 0; i < num; ++i) { interval(i) = i; }

    const Real step = delta / div;

    NEON_ASSERT_ERROR(step > 0, "STEP FUNCTION INVALID CHECK DELTA AND DIV");

    interval *= step;
    interval += (VectorXr::Ones(interval.rows()) * start);

    interval(interval.rows() - 1) = stop;

    return interval;
}
