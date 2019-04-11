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

__device__ inline double Arcsinh(double x) { return std::asinh(x); }

__device__ inline float Arcsinh(float x) { return std::asinhf(x); }

__device__ inline cuda::Float16 Arcsinh(cuda::Float16 x) { return cuda::Float16{std::asinhf(static_cast<float>(x))}; }

__device__ inline double Arccosh(double x) { return std::acosh(x); }

__device__ inline float Arccosh(float x) { return std::acoshf(x); }

__device__ inline cuda::Float16 Arccosh(cuda::Float16 x) { return cuda::Float16{std::acoshf(static_cast<float>(x))}; }

#define DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(name, func) \
    template <typename T>                              \
    __device__ inline T name(T x) {                    \
        return func(x);                                \
    }                                                  \
    __device__ inline cuda::Float16 name(cuda::Float16 x) { return cuda::Float16{func(static_cast<float>(x))}; }

DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Ceil, std::ceil)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Floor, std::floor)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Sinh, std::sinh)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Cosh, std::cosh)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Tanh, std::tanh)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Sin, std::sin)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Cos, std::cos)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Tan, std::tan)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Arcsin, std::asin)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Arccos, std::acos)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Arctan, std::atan)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Exp, std::exp)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Log, std::log)
DEFINE_CUDA_FLOAT16_FALLBACK_UNARY(Sqrt, std::sqrt)

}  // namespace cuda
}  // namespace chainerx
