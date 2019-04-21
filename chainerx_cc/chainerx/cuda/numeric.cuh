#pragma once

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
__device__ inline bool IsFinite(T /*value*/) {
    return true;
}

__device__ inline bool IsFinite(cuda::Float16 value) { return (!value.IsInf()) && (!value.IsNan()); }
__device__ inline bool IsFinite(double value) { return isfinite(value); }
__device__ inline bool IsFinite(float value) { return isfinite(value); }

__device__ inline double Arcsinh(double x) { return std::asinh(x); }

__device__ inline float Arcsinh(float x) { return std::asinhf(x); }

__device__ inline cuda::Float16 Arcsinh(cuda::Float16 x) { return cuda::Float16{std::asinhf(static_cast<float>(x))}; }

__device__ inline double Arccosh(double x) { return std::acosh(x); }

__device__ inline float Arccosh(float x) { return std::acoshf(x); }

__device__ inline cuda::Float16 Arccosh(cuda::Float16 x) { return cuda::Float16{std::acoshf(static_cast<float>(x))}; }

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
CHAINERX_DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Sqrt, std::sqrt)

}  // namespace cuda
}  // namespace chainerx
