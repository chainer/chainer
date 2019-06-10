#pragma once

#include <cstdint>
#include <type_traits>

#include "chainerx/cuda/float16.cuh"

namespace chainerx {
namespace cuda {
namespace cast_detail {

template <typename T>
constexpr bool IsFloatingPointV = std::is_floating_point<T>::value || std::is_same<std::remove_const_t<T>, cuda::Float16>::value;
}

// If float value is directly casted to unsigned, it's representation would be different from that of native.
// In order to avoid that, it's casted to int64_t first.
template <typename To, typename From>
__device__ auto cuda_numeric_cast(From from)
        -> std::enable_if_t<!std::is_same<To, bool>::value && std::is_unsigned<To>::value && cast_detail::IsFloatingPointV<From>, To> {
    return static_cast<To>(static_cast<int64_t>(from));
}

template <typename To, typename From>
__device__ auto cuda_numeric_cast(From from) ->
        // TODO(niboshi): true && is needed to pass compilation.
        std::enable_if_t<
                std::is_same<To, bool>::value || !std::is_unsigned<To>::value || !(true && cast_detail::IsFloatingPointV<From>),
                To> {
    return static_cast<To>(from);
}

}  // namespace cuda
}  // namespace chainerx
