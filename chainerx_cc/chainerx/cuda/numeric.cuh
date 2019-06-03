#pragma once
#include <type_traits>

#include "chainerx/cuda/float16.cuh"
#include "chainerx/numeric.h"

namespace chainerx {
namespace cuda {

template <typename T>
__device__ inline bool IsNan(T /*value*/) {
    return false;
}
__device__ inline bool IsNan(cuda::Float16 value) { return value.IsNan(); }
__device__ inline bool IsNan(double value) { return isnan(value); }
__device__ inline bool IsNan(float value) { return isnan(value); }

template <typename T>
__device__ inline bool IsInf(T /*value*/) {
    return false;
}
__device__ inline bool IsInf(cuda::Float16 value) { return value.IsInf(); }
__device__ inline bool IsInf(double value) { return isinf(value); }
__device__ inline bool IsInf(float value) { return isinf(value); }

template <typename T>
__device__ inline T Arctan2(T x1, T x2) {
    return std::atan2(x1, x2);
}
template <>
__device__ inline cuda::Float16 Arctan2<cuda::Float16>(cuda::Float16 x1, cuda::Float16 x2) {
    return cuda::Float16{std::atan2(static_cast<float>(x1), static_cast<float>(x2))};
}

__device__ inline double Arcsinh(double x) { return std::asinh(x); }

__device__ inline float Arcsinh(float x) { return std::asinhf(x); }

__device__ inline cuda::Float16 Arcsinh(cuda::Float16 x) { return cuda::Float16{std::asinhf(static_cast<float>(x))}; }

__device__ inline double Arccosh(double x) { return std::acosh(x); }

__device__ inline float Arccosh(float x) { return std::acoshf(x); }

__device__ inline cuda::Float16 Arccosh(cuda::Float16 x) { return cuda::Float16{std::acoshf(static_cast<float>(x))}; }

__device__ inline double Log1p(double x) { return std::log1p(x); }

__device__ inline float Log1p(float x) { return std::log1pf(x); }

__device__ inline cuda::Float16 Log1p(cuda::Float16 x) { return cuda::Float16{std::log1pf(static_cast<float>(x))}; }

template <typename T>
__device__ inline T Sign(T x) {
    return IsNan(x) ? x : static_cast<T>(static_cast<int>(T{0} < x) - static_cast<int>(x < T{0}));
}

template <>
__device__ inline uint8_t Sign(uint8_t x) {
    return static_cast<uint8_t>(x > 0);
}
template <>
__device__ inline cuda::Float16 Sign(cuda::Float16 x) {
    return IsNan(x) ? x : cuda::Float16{static_cast<int>(cuda::Float16{0} < x) - static_cast<int>(x < cuda::Float16{0})};
}

__device__ inline double Expm1(double x) { return std::expm1(x); }

__device__ inline float Expm1(float x) { return std::expm1f(x); }

__device__ inline cuda::Float16 Expm1(cuda::Float16 x) { return cuda::Float16{std::expm1f(static_cast<float>(x))}; }

__device__ inline double Exp2(double x) { return std::exp2(x); }

__device__ inline float Exp2(float x) { return std::exp2f(x); }

__device__ inline cuda::Float16 Exp2(cuda::Float16 x) { return cuda::Float16{std::exp2f(static_cast<float>(x))}; }

#define CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(name, func) \
    template <typename T>                                       \
    __device__ inline T name(T x) {                             \
        return func(x);                                         \
    }                                                           \
    __device__ inline cuda::Float16 name(cuda::Float16 x) { return cuda::Float16{func(static_cast<float>(x))}; }

CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Ceil, std::ceil)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Floor, std::floor)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Sinh, std::sinh)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Cosh, std::cosh)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Tanh, std::tanh)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Sin, std::sin)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Cos, std::cos)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Tan, std::tan)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Arcsin, std::asin)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Arccos, std::acos)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Arctan, std::atan)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Exp, std::exp)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Log, std::log)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Log10, std::log10)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Sqrt, std::sqrt)
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Fabs, std::fabs)

namespace numeric_detail {

template <typename T>
__device__ inline T NonNegativePower(T x1, T x2) {
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
__device__ inline auto Power(T x1, T x2) -> std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, T> {
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
__device__ inline auto Power(T x1, T x2) -> std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, T> {
    return numeric_detail::NonNegativePower(x1, x2);
}

template <typename T>
__device__ inline auto Power(T x1, T x2) -> std::enable_if_t<!std::is_integral<T>::value, T>;
template <>
__device__ inline cuda::Float16 Power(cuda::Float16 x1, cuda::Float16 x2) {
    return cuda::Float16{powf(static_cast<float>(x1), static_cast<float>(x2))};
}
template <>
__device__ inline float Power(float x1, float x2) {
    return powf(x1, x2);
}
template <>
__device__ inline double Power(double x1, double x2) {
    return pow(x1, x2);
}

}  // namespace cuda
}  // namespace chainerx
