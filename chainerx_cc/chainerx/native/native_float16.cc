#include "chainerx/native/native_float16.h"

#include <cstdint>

namespace chainerx {
namespace native {
namespace {
union union_float_uint {
public:
    union_float_uint(float v) : f(v) {}
    union_float_uint(uint32_t v) : i(v) {}
    float f;
    uint32_t i;
};

union union_double_uint {
public:
    union_double_uint(double v) : f(v) {}
    union_double_uint(uint64_t v) : i(v) {}
    double f;
    uint64_t i;
};

// Borrowed from npy_floatbits_to_halfbits
//
// See LICENSE.txt of ChainerX.
uint16_t floatbits_to_halfbits(uint32_t f) {
    uint32_t f_exp, f_sig;
    uint16_t h_sgn, h_exp, h_sig;

    h_sgn = static_cast<uint16_t>((f & 0x80000000u) >> 16);
    f_exp = (f & 0x7f800000u);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /* Inf or NaN */
            f_sig = (f & 0x007fffffu);
            if (f_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                uint16_t ret = static_cast<uint16_t>(0x7c00u + (f_sig >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return static_cast<uint16_t>(h_sgn + 0x7c00u);
            }
        } else {
            /* overflow to signed inf */
            return static_cast<uint16_t>(h_sgn + 0x7c00u);
        }
    }

    /* Exponent underflow converts to a subnormal half or signed zero */
    if (f_exp <= 0x38000000u) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero half-floats.
         */
        if (f_exp < 0x33000000u) {
            return h_sgn;
        }
        /* Make the subnormal significand */
        f_exp >>= 23;
        f_sig = (0x00800000u + (f & 0x007fffffu));
        f_sig >>= (113 - f_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
        f_sig += 0x00001000u;
        h_sig = static_cast<uint16_t>(f_sig >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return static_cast<uint16_t>(h_sgn + h_sig);
    }

    /* Regular case with no overflow or underflow */
    h_exp = static_cast<uint16_t>((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_sig = (f & 0x007fffffu);
    f_sig += 0x00001000u;
    h_sig = static_cast<uint16_t>(f_sig >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
    return h_sgn + h_exp + h_sig;
}

// Borrowed from npy_doublebits_to_halfbits
//
// See LICENSE.txt of ChainerX.
uint16_t doublebits_to_halfbits(uint64_t d) {
    uint64_t d_exp, d_sig;
    uint16_t h_sgn, h_exp, h_sig;

    h_sgn = (d & 0x8000000000000000ULL) >> 48;
    d_exp = (d & 0x7ff0000000000000ULL);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (d_exp >= 0x40f0000000000000ULL) {
        if (d_exp == 0x7ff0000000000000ULL) {
            /* Inf or NaN */
            d_sig = (d & 0x000fffffffffffffULL);
            if (d_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                uint16_t ret = static_cast<uint16_t>(0x7c00u + (d_sig >> 42));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return h_sgn + 0x7c00u;
            }
        } else {
            /* overflow to signed inf */
            return h_sgn + 0x7c00u;
        }
    }

    /* Exponent underflow converts to subnormal half or signed zero */
    if (d_exp <= 0x3f00000000000000ULL) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero half-floats.
         */
        if (d_exp < 0x3e60000000000000ULL) {
            return h_sgn;
        }
        /* Make the subnormal significand */
        d_exp >>= 52;
        d_sig = (0x0010000000000000ULL + (d & 0x000fffffffffffffULL));
        d_sig >>= (1009 - d_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
        d_sig += 0x0000020000000000ULL;
        h_sig = static_cast<uint16_t>(d_sig >> 42);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return h_sgn + h_sig;
    }

    /* Regular case with no overflow or underflow */
    h_exp = static_cast<uint16_t>((d_exp - 0x3f00000000000000ULL) >> 42);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    d_sig = (d & 0x000fffffffffffffULL);
    d_sig += 0x0000020000000000ULL;
    h_sig = static_cast<uint16_t>(d_sig >> 42);

    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
    return h_sgn + h_exp + h_sig;
}

// Borrowed from npy_halfbits_to_floatbits
//
// See LICENSE.txt of ChainerX.
uint32_t halfbits_to_floatbits(uint16_t h) {
    uint16_t h_exp, h_sig;
    uint32_t f_sgn, f_exp, f_sig;

    h_exp = (h & 0x7c00u);
    f_sgn = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    switch (h_exp) {
        case 0x0000u: /* 0 or subnormal */
            h_sig = (h & 0x03ffu);
            /* Signed zero */
            if (h_sig == 0) {
                return f_sgn;
            }
            /* Subnormal */
            h_sig <<= 1;
            while ((h_sig & 0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            f_exp = (static_cast<uint32_t>(127 - 15 - h_exp)) << 23;
            f_sig = (static_cast<uint32_t>(h_sig & 0x03ffu)) << 13;
            return f_sgn + f_exp + f_sig;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the significand */
            return f_sgn + 0x7f800000u + ((static_cast<uint32_t>(h & 0x03ffu)) << 13);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return f_sgn + ((static_cast<uint32_t>(h & 0x7fffu) + 0x1c000u) << 13);
    }
}

// Borrowed from npy_halfbits_to_doublebits
//
// See LICENSE.txt of ChainerX.
uint64_t halfbits_to_doublebits(uint16_t h) {
    uint16_t h_exp, h_sig;
    uint64_t d_sgn, d_exp, d_sig;

    h_exp = (h & 0x7c00u);
    d_sgn = (static_cast<uint64_t>(h) & 0x8000u) << 48;
    switch (h_exp) {
        case 0x0000u: /* 0 or subnormal */
            h_sig = (h & 0x03ffu);
            /* Signed zero */
            if (h_sig == 0) {
                return d_sgn;
            }
            /* Subnormal */
            h_sig <<= 1;
            while ((h_sig & 0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            d_exp = (static_cast<uint64_t>(1023 - 15 - h_exp)) << 52;
            d_sig = (static_cast<uint64_t>(h_sig & 0x03ffu)) << 42;
            return d_sgn + d_exp + d_sig;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the significand */
            return d_sgn + 0x7ff0000000000000ULL + ((static_cast<uint64_t>(h & 0x03ffu)) << 42);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return d_sgn + ((static_cast<uint64_t>(h & 0x7fffu) + 0xfc000u) << 42);
    }
}

uint16_t float_to_half(float v) { return floatbits_to_halfbits(union_float_uint(v).i); }
uint16_t double_to_half(double v) { return doublebits_to_halfbits(union_double_uint(v).i); }
float half_to_float(uint16_t v) { return union_float_uint(halfbits_to_floatbits(v)).f; }
double half_to_double(uint16_t v) { return union_double_uint(halfbits_to_doublebits(v)).f; }

}  // namespace

half::half(float v) : data_(float_to_half(v)) {}
half::half(double v) : data_(double_to_half(v)) {}
half::half(bool v) : data_(float_to_half(static_cast<float>(v))) {}
half::half(int16_t v) : data_(float_to_half(static_cast<float>(v))) {}
half::half(uint16_t v) : data_(float_to_half(static_cast<float>(v))) {}
half::half(int32_t v) : data_(float_to_half(static_cast<double>(v))) {}
half::half(uint32_t v) : data_(float_to_half(static_cast<double>(v))) {}
half::half(int64_t v) : data_(float_to_half(static_cast<double>(v))) {}
half::half(uint64_t v) : data_(float_to_half(static_cast<double>(v))) {}

half::operator float() const { return half_to_float(data_); }
half::operator double() const { return half_to_double(data_); }

}  // namespace native
}  // namespace chainerx
