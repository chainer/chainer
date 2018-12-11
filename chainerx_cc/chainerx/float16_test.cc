#include "chainerx/float16.h"

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace chainerx {
namespace {

void CheckNear(double expected, double actual, double tol) {
    if (std::isnan(expected)) {
        // NaN
        EXPECT_TRUE(std::isnan(actual));
    } else if (std::isinf(expected)) {
        // Signed inf
        EXPECT_EQ(expected, actual);
    } else {
        // Absolute error or relative error should be less or equal to tol.
        tol = std::max(tol, tol * std::abs(expected));
        EXPECT_NEAR(expected, actual, tol);
    }
}

void CheckFloat16(double x, double tol) {
    CheckNear(x, static_cast<float>(Half{static_cast<float>(x)}), tol);
    CheckNear(x, static_cast<float>(Half{static_cast<double>(x)}), tol);
    CheckNear(x, static_cast<double>(Half{static_cast<float>(x)}), tol);
    CheckNear(x, static_cast<double>(Half{static_cast<double>(x)}), tol);
}

TEST(NativeFloat16Test, Float16Zero) {
    CheckFloat16(0.0, 0.0);
    CheckFloat16(-0.0, 0.0);
}

TEST(NativeFloat16Test, Float16Normalized) {
    for (double x = 1e-3; x < 1e3; x *= 1.01) {
        CheckFloat16(x, 1e-3);
        CheckFloat16(-x, 1e-3);
    }
}

TEST(NativeFloat16Test, Float16Denormalized) {
    for (double x = 1e-7; x < 1e-5; x += 1e-7) {
        CheckFloat16(x, 1e-7);
        CheckFloat16(-x, 1e-7);
    }
}

TEST(NativeFloat16Test, Float16Inf) {
    CheckFloat16(std::numeric_limits<double>::infinity(), 0);
    CheckFloat16(-std::numeric_limits<double>::infinity(), 0);
}

TEST(NativeFloat16Test, Float16Nan) { CheckFloat16(NAN, 0); }

TEST(NativeFloat16Test, ToFloat16FromFloat16Idempotent) {
    for (double x = 1e-100; x < 1e100; x *= 1.01) {
        double f = static_cast<double>(Half{x});
        CheckFloat16(f, 0);
        CheckFloat16(-f, 0);
    }
}

std::vector<Half> GetFloat16Values() {
    std::vector<Half> float_values;
    float_values.emplace_back(0.0);
    for (double x = 1e-3; x < 1e2; x *= 1.05) {
        float_values.emplace_back(x);
        float_values.emplace_back(-x);
    }
    for (double x = 1e-7; x < 1e-5; x += 1e-7) {
        float_values.emplace_back(x);
        float_values.emplace_back(-x);
    }
    return float_values;
}

TEST(NativeFloat16Test, Float16Neg) {
    for (const Half& x : GetFloat16Values()) {
        CheckNear(-static_cast<double>(x), static_cast<double>(-x), 1e-3);
    }
}

TEST(NativeFloat16Test, Float16Add) {
    for (const Half& x : GetFloat16Values()) {
        for (const Half& y : GetFloat16Values()) {
            CheckNear(static_cast<double>(x) + static_cast<double>(y), static_cast<double>(x + y), 1e-3);
        }
    }
}

TEST(NativeFloat16Test, Float16Subtract) {
    for (const Half& x : GetFloat16Values()) {
        for (const Half& y : GetFloat16Values()) {
            CheckNear(static_cast<double>(x) - static_cast<double>(y), static_cast<double>(x - y), 1e-3);
        }
    }
}

TEST(NativeFloat16Test, Float16Multiply) {
    for (const Half& x : GetFloat16Values()) {
        for (const Half& y : GetFloat16Values()) {
            CheckNear(static_cast<double>(x) * static_cast<double>(y), static_cast<double>(x * y), 1e-3);
        }
    }
}

TEST(NativeFloat16Test, Float16Divide) {
    for (const Half& x : GetFloat16Values()) {
        for (const Half& y : GetFloat16Values()) {
            double denom = static_cast<double>(y);
            if (-1e-2 < denom && denom < 1e-2) {
                continue;
            }
            CheckNear(static_cast<double>(x) / denom, static_cast<double>(x / y), 1e-3);
        }
    }
}

TEST(NativeFloat16Test, Float16AddI) {
    for (const Half& x : GetFloat16Values()) {
        for (const Half& y : GetFloat16Values()) {
            double expected = static_cast<double>(x) + static_cast<double>(y);
            Half x_copy = x;
            CheckNear(expected, static_cast<double>(x_copy += y), 1e-3);
        }
    }
}

TEST(NativeFloat16Test, Float16SubtractI) {
    for (const Half& x : GetFloat16Values()) {
        for (const Half& y : GetFloat16Values()) {
            double expected = static_cast<double>(x) - static_cast<double>(y);
            Half x_copy = x;
            CheckNear(expected, static_cast<double>(x_copy -= y), 1e-3);
        }
    }
}

TEST(NativeFloat16Test, Float16MultiplyI) {
    for (const Half& x : GetFloat16Values()) {
        for (const Half& y : GetFloat16Values()) {
            double expected = static_cast<double>(x) * static_cast<double>(y);
            Half x_copy = x;
            CheckNear(expected, static_cast<double>(x_copy *= y), 1e-3);
        }
    }
}

TEST(NativeFloat16Test, Float16DivideI) {
    for (const Half& x : GetFloat16Values()) {
        for (const Half& y : GetFloat16Values()) {
            double denom = static_cast<double>(y);
            if (-1e-2 < denom && denom < 1e-2) {
                continue;
            }
            double expected = static_cast<double>(x) / denom;
            Half x_copy = x;
            CheckNear(expected, static_cast<double>(x_copy /= y), 1e-3);
        }
    }
}

}  // namespace
}  // namespace chainerx
