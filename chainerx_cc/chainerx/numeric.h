#pragma once
#include <cmath>
#include <type_traits>

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

bool AllClose(const Array& a, const Array& b, double rtol = 1e-5, double atol = 1e-8, bool equal_nan = false);

template <typename T>
inline bool IsNan(T /*value*/) {
    return false;
}

inline bool IsNan(chainerx::Float16 value) { return value.IsNan(); }
inline bool IsNan(float value) { return std::isnan(value); }
inline bool IsNan(double value) { return std::isnan(value); }

template <typename T>
inline bool IsInf(T /*value*/) {
    return false;
}

inline bool IsInf(chainerx::Float16 value) { return value.IsInf(); }
inline bool IsInf(double value) { return std::isinf(value); }
inline bool IsInf(float value) { return std::isinf(value); }

template <typename T>
inline T Sign(T x) {
    return IsNan(x) ? x : static_cast<T>(static_cast<int>(T{0} < x) - static_cast<int>(x < T{0}));
}

template <>
inline chainerx::Float16 Sign<chainerx::Float16>(chainerx::Float16 x) {
    return IsNan(x) ? x : Float16{static_cast<int>(Float16{0} < x) - static_cast<int>(x < Float16{0})};
}

#define CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(name, func)           \
    template <typename T>                                                   \
    inline T name(T x) {                                                    \
        return func(x);                                                     \
    }                                                                       \
    template <>                                                             \
    inline chainerx::Float16 name<chainerx::Float16>(chainerx::Float16 x) { \
        return chainerx::Float16{func(static_cast<float>(x))};              \
    }

CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Ceil, std::ceil)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Floor, std::floor)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Sinh, std::sinh)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Cosh, std::cosh)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Tanh, std::tanh)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Arcsinh, std::asinh)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Arccosh, std::acosh)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Sin, std::sin)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Cos, std::cos)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Tan, std::tan)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Arcsin, std::asin)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Arccos, std::acos)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Arctan, std::atan)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Exp, std::exp)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Expm1, std::expm1)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Exp2, std::exp2)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Log, std::log)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Log10, std::log10)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Log1p, std::log1p)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Sqrt, std::sqrt)
CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Fabs, std::fabs)

namespace numeric_detail {

template <typename T>
inline T NonNegativePower(T x1, T x2) {
    static_assert(std::is_integral<T>::value, "NonNegativePower is only defined for non-negative integrals.");
    T out{1};

    while (x2 > 0) {
        if (x2 & 1) {
            out *= x1;
        }
        x1 *= x1;
        x2 >>= 1;
    }

    return out;
}

}  // namespace numeric_detail

template <typename T>
inline auto Power(T x1, T x2) -> std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, T> {
    if (x2 < 0) {
        switch (x1) {
            case -1:
                return x2 & 1 ? -1 : 1;
            case 1:
                return 1;
            default:
                return 0;
        }
    }
    return numeric_detail::NonNegativePower(x1, x2);
}

template <typename T>
inline auto Power(T x1, T x2) -> std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, T> {
    return numeric_detail::NonNegativePower(x1, x2);
}

template <typename T>
inline auto Power(T x1, T x2) -> std::enable_if_t<!std::is_integral<T>::value, T>;
template <>
inline chainerx::Float16 Power(chainerx::Float16 x1, chainerx::Float16 x2) {
    return chainerx::Float16{std::pow(static_cast<float>(x1), static_cast<float>(x2))};
}
template <>
inline float Power(float x1, float x2) {
    return std::pow(x1, x2);
}
template <>
inline double Power(double x1, double x2) {
    return std::pow(x1, x2);
}

#define CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_BINARY(name, func)                                 \
    template <typename T>                                                                          \
    inline T name(T x1, T x2) {                                                                    \
        return func(x1, x2);                                                                       \
    }                                                                                              \
    template <>                                                                                    \
    inline chainerx::Float16 name<chainerx::Float16>(chainerx::Float16 x1, chainerx::Float16 x2) { \
        return chainerx::Float16{func(static_cast<float>(x1), static_cast<float>(x2))};            \
    }

CHAINERX_DEFINE_NATIVE_FLOAT16_FALLBACK_BINARY(Arctan2, std::atan2)

}  // namespace chainerx
