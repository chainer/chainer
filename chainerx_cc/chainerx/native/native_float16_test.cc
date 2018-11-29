#include "chainerx/native/native_float16.h"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace chainerx {
namespace native {
namespace {

void CheckNear(double expected, double actual, double tol) {
    if (std::isinf(expected)) {
        EXPECT_TRUE(std::isinf(actual));
    } else if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual));
    } else if (expected == 0.0) {
        EXPECT_EQ(actual, 0.0);
    } else {
        EXPECT_NE(actual, 0.0);
        EXPECT_NEAR(expected, actual, tol);
    }
}

void CheckFloatSixteen(double x, double tol) {
    CheckNear(x, static_cast<float>(half(static_cast<float>(x))), tol);
    CheckNear(x, static_cast<float>(half(static_cast<double>(x))), tol);
    CheckNear(x, static_cast<double>(half(static_cast<float>(x))), tol);
    CheckNear(x, static_cast<double>(half(static_cast<double>(x))), tol);
}

TEST(NativeFloatSixteenTest, FloatSixteenNormalized) {
    CheckFloatSixteen(0, 0);
    for (double x = 1e-3; x < 30; x *= 1.01) {
        CheckFloatSixteen(x, 1e-2);
        CheckFloatSixteen(-x, 1e-2);
    }
}

TEST(NativeFloatSixteenTest, FloatSixteenDenormalized) {
    for (double x = 1e-7; x < 1e-5; x += 1e-7) {
        CheckFloatSixteen(x, 1e-7);
        CheckFloatSixteen(-x, 1e-7);
    }
}

TEST(NativeFloatSixteenTest, FloatSixteenInf) {
    CheckFloatSixteen(std::numeric_limits<double>::infinity(), 0);
    CheckFloatSixteen(-std::numeric_limits<double>::infinity(), 0);
}

TEST(NativeFloatSixteenTest, FloatSixteenNan) { CheckFloatSixteen(NAN, 0); }

TEST(NativeFloatSixteenTest, ToFloatSixteenFromFloatSixteenIdempotent) {
    for (double x = 1e-100; x < 1e100; x *= 1.01) {
        double f = static_cast<double>(static_cast<half>(x));
        CheckFloatSixteen(f, 0);
        CheckFloatSixteen(-f, 0);
    }
}

}  // namespace
}  // namespace native
}  // namespace chainerx
