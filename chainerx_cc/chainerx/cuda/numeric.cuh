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
__device__ inline T Ceil(T x) {
    return std::ceil(x);
}

__device__ inline cuda::Float16 Ceil(cuda::Float16 x) { return cuda::Float16{std::ceil(static_cast<float>(x))}; }

template <typename T>
__device__ inline T Tanh(T x) {
    return std::tanh(x);
}

__device__ inline cuda::Float16 Tanh(cuda::Float16 x) { return cuda::Float16{std::tanh(static_cast<float>(x))}; }

template <typename T>
__device__ inline T Sin(T x) {
    return std::sin(x);
}

__device__ inline cuda::Float16 Sin(cuda::Float16 x) { return cuda::Float16{std::sin(static_cast<float>(x))}; }

template <typename T>
__device__ inline T Cos(T x) {
    return std::cos(x);
}

__device__ inline cuda::Float16 Cos(cuda::Float16 x) { return cuda::Float16{std::cos(static_cast<float>(x))}; }

template <typename T>
__device__ inline T Tan(T x) {
    return std::tan(x);
}

__device__ inline cuda::Float16 Tan(cuda::Float16 x) { return cuda::Float16{std::tan(static_cast<float>(x))}; }

template <typename T>
__device__ inline T Arcsin(T x) {
    return std::asin(x);
}

__device__ inline cuda::Float16 Arcsin(cuda::Float16 x) { return cuda::Float16{std::asin(static_cast<float>(x))}; }

template <typename T>
__device__ inline T Arccos(T x) {
    return std::acos(x);
}

__device__ inline cuda::Float16 Arccos(cuda::Float16 x) { return cuda::Float16{std::acos(static_cast<float>(x))}; }

template <typename T>
__device__ inline T Arctan(T x) {
    return std::atan(x);
}

__device__ inline cuda::Float16 Arctan(cuda::Float16 x) { return cuda::Float16{std::atan(static_cast<float>(x))}; }

template <typename T>
__device__ inline T Exp(T x) {
    return std::exp(x);
}

__device__ inline cuda::Float16 Exp(cuda::Float16 x) { return x.Exp(); }

template <typename T>
__device__ inline T Log(T x) {
    return std::log(x);
}

__device__ inline cuda::Float16 Log(cuda::Float16 x) { return x.Log(); }

template <typename T>
__device__ inline T Sqrt(T x) {
    return std::sqrt(x);
}

__device__ inline cuda::Float16 Sqrt(cuda::Float16 x) { return x.Sqrt(); }

}  // namespace cuda
}  // namespace chainerx
