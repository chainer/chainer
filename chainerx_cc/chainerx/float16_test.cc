#include "chainerx/float16.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace chainerx {
namespace {

bool IsNan(Float16 x) {
    uint16_t exp = x.data() & 0x7c00U;
    uint16_t frac = x.data() & 0x03ffU;
    return exp == 0x7c00U && frac != 0x0000U;
}

// Checks if `d` is equal to FromFloat16(ToFloat16(d)) with tolerance `tol`.
// This function cannot take NaN as a parameter.  The cast of NaN is tested in `Float16Nan`.
void CheckToFloat16FromFloat16Near(double d, double tol) {
    Float16 h{d};
    Float16 f{static_cast<float>(d)};
    EXPECT_EQ(h.data(), f.data());
    float f_result = static_cast<float>(h);
    double d_result = static_cast<double>(h);

    ASSERT_FALSE(std::isnan(d));
    EXPECT_FALSE(std::isnan(f_result));
    EXPECT_FALSE(std::isnan(d_result));
    EXPECT_FALSE(IsNan(h));

    if (std::isinf(d)) {
        // Signed inf
        EXPECT_EQ(d, f_result);
        EXPECT_EQ(d, d_result);
    } else {
        // Absolute error or relative error should be less or equal to tol.
        tol = std::max(tol, tol * std::abs(d));
        EXPECT_NEAR(d, f_result, tol);
        EXPECT_NEAR(d, d_result, tol);
    }
}

// Checks if `h` is equal to ToFloat16(FromFloat16(h)) exactly.
// This function cannot take NaN as a parameter.  The cast of NaN is tested in `Float16Nan`.
void CheckFromFloat16ToFloat16Eq(Float16 h) {
    float f = static_cast<float>(h);
    double d = static_cast<double>(h);
    EXPECT_EQ(d, static_cast<double>(f));

    ASSERT_FALSE(IsNan(h));
    EXPECT_FALSE(std::isnan(f));
    EXPECT_FALSE(std::isnan(d));

    EXPECT_EQ(h.data(), Float16{f}.data());
    EXPECT_EQ(h.data(), Float16{d}.data());
}

TEST(NativeFloat16Test, Float16Zero) {
    EXPECT_EQ(Float16{float{0.0}}.data(), 0x0000U);
    EXPECT_EQ(Float16{float{-0.0}}.data(), 0x8000U);
    EXPECT_EQ(Float16{double{0.0}}.data(), 0x0000U);
    EXPECT_EQ(Float16{double{-0.0}}.data(), 0x8000U);
    EXPECT_EQ(static_cast<float>(Float16::FromData(0x0000U)), 0.0);
    EXPECT_EQ(static_cast<float>(Float16::FromData(0x8000U)), -0.0);
    EXPECT_EQ(static_cast<double>(Float16::FromData(0x0000U)), 0.0);
    EXPECT_EQ(static_cast<double>(Float16::FromData(0x8000U)), -0.0);
    // Checks if the value is casted to 0.0 or -0.0
    EXPECT_EQ(1 / static_cast<float>(Float16::FromData(0x0000U)), std::numeric_limits<float>::infinity());
    EXPECT_EQ(1 / static_cast<float>(Float16::FromData(0x8000U)), -std::numeric_limits<float>::infinity());
    EXPECT_EQ(1 / static_cast<double>(Float16::FromData(0x0000U)), std::numeric_limits<float>::infinity());
    EXPECT_EQ(1 / static_cast<double>(Float16::FromData(0x8000U)), -std::numeric_limits<float>::infinity());
}

TEST(NativeFloat16Test, Float16Normalized) {
    for (double x = 1e-3; x < 1e3; x *= 1.01) {  // NOLINT(clang-analyzer-security.FloatLoopCounter,cert-flp30-c)
        EXPECT_NE(Float16{x}.data() & 0x7c00U, 0U);
        CheckToFloat16FromFloat16Near(x, 1e-3);
        CheckToFloat16FromFloat16Near(-x, 1e-3);
    }
    for (uint16_t bit = 0x0400; bit < 0x7c00; ++bit) {
        CheckFromFloat16ToFloat16Eq(Float16::FromData(bit | 0x0000U));
        CheckFromFloat16ToFloat16Eq(Float16::FromData(bit | 0x8000U));
    }
}

TEST(NativeFloat16Test, Float16Denormalized) {
    for (double x = 1e-7; x < 1e-5; x += 1e-7) {  // NOLINT(clang-analyzer-security.FloatLoopCounter,cert-flp30-c)
        // Check if the underflow gap around zero is filled with denormal number.
        EXPECT_EQ(Float16{x}.data() & 0x7c00U, 0x0000U);
        EXPECT_NE(Float16{x}.data() & 0x03ffU, 0x0000U);
        CheckToFloat16FromFloat16Near(x, 1e-7);
        CheckToFloat16FromFloat16Near(-x, 1e-7);
    }
    for (uint16_t bit = 0x0000U; bit < 0x0400U; ++bit) {
        CheckFromFloat16ToFloat16Eq(Float16::FromData(bit | 0x0000U));
        CheckFromFloat16ToFloat16Eq(Float16::FromData(bit | 0x8000U));
    }
}

TEST(NativeFloat16Test, Float16Inf) {
    EXPECT_EQ(Float16{std::numeric_limits<float>::infinity()}.data(), 0x7c00U);
    EXPECT_EQ(Float16{-std::numeric_limits<float>::infinity()}.data(), 0xfc00U);
    EXPECT_EQ(Float16{std::numeric_limits<double>::infinity()}.data(), 0x7c00U);
    EXPECT_EQ(Float16{-std::numeric_limits<double>::infinity()}.data(), 0xfc00U);
    EXPECT_EQ(std::numeric_limits<float>::infinity(), static_cast<float>(Float16::FromData(0x7c00U)));
    EXPECT_EQ(-std::numeric_limits<float>::infinity(), static_cast<float>(Float16::FromData(0xfc00U)));
    EXPECT_EQ(std::numeric_limits<double>::infinity(), static_cast<double>(Float16::FromData(0x7c00U)));
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), static_cast<double>(Float16::FromData(0xfc00U)));
}

TEST(NativeFloat16Test, Float16Nan) {
    for (uint16_t bit = 0x7c01U; bit < 0x8000U; ++bit) {
        EXPECT_TRUE(std::isnan(static_cast<float>(Float16::FromData(bit | 0x0000U))));
        EXPECT_TRUE(std::isnan(static_cast<float>(Float16::FromData(bit | 0x8000U))));
        EXPECT_TRUE(std::isnan(static_cast<double>(Float16::FromData(bit | 0x0000U))));
        EXPECT_TRUE(std::isnan(static_cast<double>(Float16::FromData(bit | 0x8000U))));
    }
    EXPECT_TRUE(IsNan(Float16{float{NAN}}));
    EXPECT_TRUE(IsNan(Float16{double{NAN}}));
}

union Float32Bits {
public:
    explicit Float32Bits(uint32_t v) : u{v} {}
    uint32_t u;
    float f;
};

// Test if a fp32 was NOT rounded as specified in numpy algorithm
TEST(NativeFloat16Test, Float16Rounding32) {
    // For the significand not to be rounded half significand must be even,
    // and the remaining pattern 1000...0
    Float32Bits bits{static_cast<uint32_t>((0x3fU << 24) + (5 << 12))};
    // If the number was not rounded, then the last bit of the
    // significand is lost and the rest remain as it was
    // (5 << 12 ) >> 13
    EXPECT_EQ((Float16{bits.f}.data() & 0x3ffU), 0x2U);
}

union Float64Bits {
public:
    explicit Float64Bits(uint64_t v) : u{v} {}
    uint64_t u;
    double f;
};

// Test if a fp64 was NOT rounded as specified in numpy algorithm
TEST(NativeFloat16Test, Float16Rounding64) {
    // For the significand not to be rounded half significand must be even,
    // and the remaining pattern 1000...0
    Float64Bits bits{static_cast<uint64_t>((0x3ffULL << 52) + (5ULL << 41))};
    // If the number was not rounded, then the last bit of the
    // significand is lost and the rest remain as it was
    // (5 << 12 ) >> 13
    EXPECT_EQ((Float16{bits.f}.data() & 0x3ffU), 0x2U);
}

// Get the partial set of all Float16 values for reduction of test execution time.
// The returned list includes the all values whose trailing 8 digits are `0b00000000` or `0b01010101`.
// This list includes all special values (e.g. signed zero, infinity) and some of normalized/denormalize numbers and NaN.
std::vector<Float16> GetFloat16Values() {
    std::vector<Float16> values;
    values.reserve(1 << 9);
    // Use uint32_t instead of uint16_t to avoid overflow
    for (uint32_t bit = 0x0000U; bit <= 0xffffU; bit += 0x0100U) {
        values.emplace_back(Float16::FromData(bit | 0x0000U));
        values.emplace_back(Float16::FromData(bit | 0x0055U));
    }
    return values;
}

// Checks if `l` is equal to `r` or both of them are NaN.
void ExpectEqFloat16(Float16 l, Float16 r) {
    if (IsNan(l) && IsNan(r)) {
        return;
    }
    EXPECT_EQ(l.data(), r.data());
}

TEST(NativeFloat16Test, Float16Neg) {
    // Use uint32_t instead of uint16_t to avoid overflow
    for (uint32_t bit = 0x0000U; bit <= 0xffffU; ++bit) {
        Float16 x = Float16::FromData(bit);
        Float16 expected{-static_cast<double>(x)};
        ExpectEqFloat16(expected, -x);
    }
}

TEST(NativeFloat16Test, Float16Add) {
    for (Float16 x : GetFloat16Values()) {
        for (Float16 y : GetFloat16Values()) {
            Float16 expected{static_cast<double>(x) + static_cast<double>(y)};
            ExpectEqFloat16(expected, x + y);
            ExpectEqFloat16(expected, y + x);
        }
    }
}

TEST(NativeFloat16Test, Float16Subtract) {
    for (Float16 x : GetFloat16Values()) {
        for (Float16 y : GetFloat16Values()) {
            Float16 expected{static_cast<double>(x) - static_cast<double>(y)};
            ExpectEqFloat16(expected, x - y);
        }
    }
}

TEST(NativeFloat16Test, Float16Multiply) {
    for (Float16 x : GetFloat16Values()) {
        for (Float16 y : GetFloat16Values()) {
            Float16 expected{static_cast<double>(x) * static_cast<double>(y)};
            ExpectEqFloat16(expected, x * y);
            ExpectEqFloat16(expected, y * x);
            EXPECT_EQ(expected.data(), (x * y).data());
        }
    }
}

TEST(NativeFloat16Test, Float16Divide) {
    for (Float16 x : GetFloat16Values()) {
        for (Float16 y : GetFloat16Values()) {
            Float16 expected{static_cast<double>(x) / static_cast<double>(y)};
            ExpectEqFloat16(expected, x / y);
        }
    }
}

TEST(NativeFloat16Test, Float16AddI) {
    for (Float16 x : GetFloat16Values()) {
        for (Float16 y : GetFloat16Values()) {
            Float16 expected{static_cast<double>(y) + static_cast<double>(x)};
            Float16 z = (y += x);
            ExpectEqFloat16(expected, y);
            ExpectEqFloat16(expected, z);
        }
    }
}

TEST(NativeFloat16Test, Float16SubtractI) {
    for (Float16 x : GetFloat16Values()) {
        for (Float16 y : GetFloat16Values()) {
            Float16 expected{static_cast<double>(y) - static_cast<double>(x)};
            Float16 z = y -= x;
            ExpectEqFloat16(expected, y);
            ExpectEqFloat16(expected, z);
        }
    }
}

TEST(NativeFloat16Test, Float16MultiplyI) {
    for (Float16 x : GetFloat16Values()) {
        for (Float16 y : GetFloat16Values()) {
            Float16 expected{static_cast<double>(y) * static_cast<double>(x)};
            Float16 z = y *= x;
            ExpectEqFloat16(expected, y);
            ExpectEqFloat16(expected, z);
        }
    }
}

TEST(NativeFloat16Test, Float16DivideI) {
    for (Float16 x : GetFloat16Values()) {
        for (Float16 y : GetFloat16Values()) {
            Float16 expected{static_cast<double>(y) / static_cast<double>(x)};
            Float16 z = y /= x;
            ExpectEqFloat16(expected, y);
            ExpectEqFloat16(expected, z);
        }
    }
}

TEST(NativeFloat16Test, FloatComparison) {
    for (Float16 x : GetFloat16Values()) {
        for (Float16 y : GetFloat16Values()) {
#define CHECK_COMPARISION_OPERATOR(op)                                          \
    {                                                                           \
        /* NOLINTNEXTLINE(misc-macro-parentheses,bugprone-macro-parentheses) */ \
        EXPECT_EQ(static_cast<double>(x) op static_cast<double>(y), x op y);    \
        /* NOLINTNEXTLINE(misc-macro-parentheses,bugprone-macro-parentheses) */ \
        EXPECT_EQ(static_cast<double>(y) op static_cast<double>(x), y op x);    \
    }

            CHECK_COMPARISION_OPERATOR(==);
            CHECK_COMPARISION_OPERATOR(!=);
            CHECK_COMPARISION_OPERATOR(<);  // NOLINT(whitespace/operators)
            CHECK_COMPARISION_OPERATOR(>);  // NOLINT(whitespace/operators)
            CHECK_COMPARISION_OPERATOR(<=);
            CHECK_COMPARISION_OPERATOR(>=);

#undef CHECK_COMPARISION_OPERATOR
        }
    }
}

}  // namespace
}  // namespace chainerx
