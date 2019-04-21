#pragma once
#include <cmath>

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

#define DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(name, func)                    \
    template <typename T>                                                   \
    inline T name(T x) {                                                    \
        return func(x);                                                     \
    }                                                                       \
    template <>                                                             \
    inline chainerx::Float16 name<chainerx::Float16>(chainerx::Float16 x) { \
        return chainerx::Float16{func(static_cast<float>(x))};              \
    }

DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Ceil, std::ceil)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Floor, std::floor)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Sinh, std::sinh)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Cosh, std::cosh)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Tanh, std::tanh)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Arcsinh, std::asinh)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Arccosh, std::acosh)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Sin, std::sin)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Cos, std::cos)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Tan, std::tan)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Arcsin, std::asin)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Arccos, std::acos)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Arctan, std::atan)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Exp, std::exp)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Log, std::log)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Log10, std::log10)
DEFINE_NATIVE_FLOAT16_FALLBACK_UNARY(Sqrt, std::sqrt)

}  // namespace chainerx
