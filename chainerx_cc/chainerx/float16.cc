#include "chainerx/float16.h"

#include <cstdint>

namespace chainerx {
namespace {

union UnionFloatUint {
public:
    explicit UnionFloatUint(float v) : f{v} {}
    explicit UnionFloatUint(uint32_t v) : i{v} {}
    float f;
    uint32_t i;
};

union UnionDoubleUint {
public:
    explicit UnionDoubleUint(double v) : f{v} {}
    explicit UnionDoubleUint(uint64_t v) : i{v} {}
    double f;
    uint64_t i;
};

// Borrowed from npy_floatbits_to_halfbits
//
// See LICENSE.txt of ChainerX.
uint16_t FloatbitsToHalfbits(uint32_t f) {
    uint16_t h_sgn = static_cast<uint16_t>((f & 0x80000000U) >> 16);
    uint32_t f_exp = (f & 0x7f800000U);

    // Exponent overflow/NaN converts to signed inf/NaN
    if (f_exp >= 0x47800000U) {
        if (f_exp != 0x7f800000U) {
            // Overflow to signed inf
            return h_sgn + 0x7c00U;
        }

        uint32_t f_sig = (f & 0x007fffffU);
        if (f_sig == 0) {
            // Signed inf
            return h_sgn + 0x7c00U;
        }

        // NaN - propagate the flag in the significand...
        uint16_t ret = static_cast<uint16_t>(0x7c00U + (f_sig >> 13));

        // ...but make sure it stays a NaN
        if (ret == 0x7c00U) {
            ++ret;
        }
        return h_sgn + ret;
    }

    // Exponent underflow converts to a subnormal half or signed zero
    if (f_exp <= 0x38000000U) {
        if (f_exp < 0x33000000U) {
            // Signed zero
            return h_sgn;
        }

        // Make the subnormal significand
        f_exp >>= 23;
        uint32_t f_sig = (0x00800000U + (f & 0x007fffffU)) >> (113 - f_exp);

        // Handle rounding by adding 1 to the bit beyond half precision
        if (((f_sig & 0x00003fffU) != 0x00001000U) || ((f & 0x000007ffU) > 0)) {
            f_sig += 0x00001000U;
        }
        uint16_t h_sig = static_cast<uint16_t>(f_sig >> 13);

        // If the rounding causes a bit to spill into h_exp, it will increment h_exp from zero to one and h_sig will be zero. This is the
        // correct result.
        return h_sgn + h_sig;
    }

    // Regular case with no overflow or underflow
    uint16_t h_exp = static_cast<uint16_t>((f_exp - 0x38000000U) >> 13);

    // Handle rounding by adding 1 to the bit beyond half precision
    uint32_t f_sig = f & 0x007fffffU;
    if ((f_sig & 0x00003fffU) != 0x00001000U) {
        f_sig += 0x00001000U;
    }
    uint16_t h_sig = static_cast<uint16_t>(f_sig >> 13);

    // If the rounding causes a bit to spill into h_exp, it will increment h_exp by one and h_sig will be zero. This is the correct result.
    // h_exp may increment to 15, at greatest, in which case the result overflows to a signed inf.
    return h_sgn + h_exp + h_sig;
}

// Borrowed from npy_doublebits_to_halfbits
//
// See LICENSE.txt of ChainerX.
uint16_t DoublebitsToHalfbits(uint64_t d) {
    uint16_t h_sgn = (d & 0x8000000000000000ULL) >> 48;
    uint64_t d_exp = (d & 0x7ff0000000000000ULL);

    // Exponent overflow/NaN converts to signed inf/NaN
    if (d_exp >= 0x40f0000000000000ULL) {
        if (d_exp != 0x7ff0000000000000ULL) {
            // Overflow to signed inf
            return h_sgn + 0x7c00U;
        }

        uint64_t d_sig = (d & 0x000fffffffffffffULL);
        if (d_sig == 0) {
            // Signed inf
            return h_sgn + 0x7c00U;
        }

        // NaN - propagate the flag in the significand...
        uint16_t ret = static_cast<uint16_t>(0x7c00U + (d_sig >> 42));

        // ...but make sure it stays a NaN
        if (ret == 0x7c00U) {
            ++ret;
        }
        return h_sgn + ret;
    }

    // Exponent underflow converts to subnormal half or signed zero
    if (d_exp <= 0x3f00000000000000ULL) {
        if (d_exp < 0x3e60000000000000ULL) {
            // Signed zero
            return h_sgn;
        }

        // Make the subnormal significand
        d_exp >>= 52;
        uint64_t d_sig = (0x0010000000000000ULL + (d & 0x000fffffffffffffULL));
        d_sig <<= (d_exp - 998);
        // Handle rounding by adding 1 to the bit beyond half precision
        if ((d_sig & 0x003fffffffffffffULL) != 0x0010000000000000ULL) {
            d_sig += 0x0010000000000000ULL;
        }
        uint16_t h_sig = static_cast<uint16_t>(d_sig >> 53);

        // If the rounding causes a bit to spill into h_exp, it will increment h_exp from zero to one and h_sig will be zero. This is the
        // correct result.
        return h_sgn + h_sig;
    }

    // Regular case with no overflow or underflow
    uint16_t h_exp = static_cast<uint16_t>((d_exp - 0x3f00000000000000ULL) >> 42);

    // Handle rounding by adding 1 to the bit beyond half precision
    uint64_t d_sig = d & 0x000fffffffffffffULL;
    if ((d_sig & 0x000007ffffffffffULL) != 0x0000020000000000ULL) {
        d_sig += 0x0000020000000000ULL;
    }
    uint16_t h_sig = static_cast<uint16_t>(d_sig >> 42);

    // If the rounding causes a bit to spill into h_exp, it will increment h_exp by one and h_sig will be zero. This is the correct result.
    // h_exp may increment to 15, at greatest, in which case the result overflows to a signed inf.
    return h_sgn + h_exp + h_sig;
}

// Borrowed from npy_halfbits_to_floatbits
//
// See LICENSE.txt of ChainerX.
uint32_t HalfbitsToFloatbits(uint16_t h) {
    uint16_t h_exp = (h & 0x7c00U);
    uint32_t f_sgn = (static_cast<uint32_t>(h) & 0x8000U) << 16;
    switch (h_exp) {
        case 0x0000U: {  // 0 or subnormal
            uint16_t h_sig = (h & 0x03ffU);

            // Signed zero
            if (h_sig == 0) {
                return f_sgn;
            }

            // Subnormal
            h_sig <<= 1;
            while ((h_sig & 0x0400U) == 0) {
                h_sig <<= 1;
                ++h_exp;
            }

            uint32_t f_exp = (static_cast<uint32_t>(127 - 15 - h_exp)) << 23;
            uint32_t f_sig = (static_cast<uint32_t>(h_sig & 0x03ffU)) << 13;
            return f_sgn + f_exp + f_sig;
        }
        case 0x7c00U: {  // inf or NaN
            // All-ones exponent and a copy of the significand
            return f_sgn + 0x7f800000U + ((static_cast<uint32_t>(h & 0x03ffU)) << 13);
        }
        default: {  // normalized
            // Just need to adjust the exponent and shift
            return f_sgn + ((static_cast<uint32_t>(h & 0x7fffU) + 0x1c000U) << 13);
        }
    }
}

// Borrowed from npy_halfbits_to_doublebits
//
// See LICENSE.txt of ChainerX.
uint64_t HalfbitsToDoublebits(uint16_t h) {
    uint16_t h_exp = (h & 0x7c00U);
    uint64_t d_sgn = (static_cast<uint64_t>(h) & 0x8000U) << 48;
    switch (h_exp) {
        case 0x0000U: {  // 0 or subnormal
            uint16_t h_sig = (h & 0x03ffU);

            // Signed zero
            if (h_sig == 0) {
                return d_sgn;
            }

            // Subnormal
            h_sig <<= 1;
            while ((h_sig & 0x0400U) == 0) {
                h_sig <<= 1;
                ++h_exp;
            }

            uint64_t d_exp = (static_cast<uint64_t>(1023 - 15 - h_exp)) << 52;
            uint64_t d_sig = (static_cast<uint64_t>(h_sig & 0x03ffU)) << 42;
            return d_sgn + d_exp + d_sig;
        }
        case 0x7c00U: {  // inf or NaN
            // All-ones exponent and a copy of the significand
            return d_sgn + 0x7ff0000000000000ULL + ((static_cast<uint64_t>(h & 0x03ffU)) << 42);
        }
        default: {  // normalized
            // Just need to adjust the exponent and shift
            return d_sgn + ((static_cast<uint64_t>(h & 0x7fffU) + 0xfc000U) << 42);
        }
    }
}

uint16_t FloatToHalf(float v) {
    return FloatbitsToHalfbits(UnionFloatUint(v).i);  // NOLINT(cppcoreguidelines-pro-type-union-access)
}
uint16_t DoubleToHalf(double v) {
    return DoublebitsToHalfbits(UnionDoubleUint(v).i);  // NOLINT(cppcoreguidelines-pro-type-union-access)
}
float HalfToFloat(uint16_t v) {
    return UnionFloatUint(HalfbitsToFloatbits(v)).f;  // NOLINT(cppcoreguidelines-pro-type-union-access)
}
double HalfToDouble(uint16_t v) {
    return UnionDoubleUint(HalfbitsToDoublebits(v)).f;  // NOLINT(cppcoreguidelines-pro-type-union-access)
}

}  // namespace

Float16::Float16(float v) : data_{FloatToHalf(v)} {}
Float16::Float16(double v) : data_{DoubleToHalf(v)} {}

Float16::operator float() const { return HalfToFloat(data_); }
Float16::operator double() const { return HalfToDouble(data_); }

}  // namespace chainerx
