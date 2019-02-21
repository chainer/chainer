#pragma once

#include "chainerx/cuda/float16.cuh"
#include "chainerx/numeric_limits.h"

namespace chainerx {
namespace cuda {

// Numerical limit values used in native and CUDA kernel implementations.
template <typename T>
struct NumericLimits {
    __host__ __device__ static constexpr T LowestOrInf() noexcept { return chainerx::NumericLimits<T>::LowestOrInf(); }
    __host__ __device__ static constexpr T MaxOrInf() noexcept { return chainerx::NumericLimits<T>::MaxOrInf(); }
};

template <>
struct NumericLimits<cuda::Float16> {
    __host__ __device__ static constexpr cuda::Float16 LowestOrInf() noexcept { return cuda::Float16::NegInf(); }
    __host__ __device__ static constexpr cuda::Float16 MaxOrInf() noexcept { return cuda::Float16::Inf(); }
};

}  // namespace cuda
}  // namespace chainerx
