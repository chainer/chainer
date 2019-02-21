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
__device__ inline T Tanh(T x) {
    return std::tanh(x);
}

__device__ inline cuda::Float16 Tanh(cuda::Float16 x) { return cuda::Float16{std::tanh(static_cast<float>(x))}; }

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
