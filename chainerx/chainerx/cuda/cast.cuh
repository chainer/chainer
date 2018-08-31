#pragma once

#include <cstdint>
#include <type_traits>

namespace chainerx {
namespace cuda {

// If float value is directly casted to unsigned, it's representation would be different from that of native.
// In order to avoid that, it's casted to int64_t first.
template <typename To, typename From>
__device__ auto cuda_numeric_cast(From from)
        -> std::enable_if_t<!std::is_same<To, bool>::value && std::is_unsigned<To>::value && std::is_floating_point<From>::value, To> {
    return static_cast<To>(static_cast<int64_t>(from));
}

template <typename To, typename From>
__device__ auto cuda_numeric_cast(From from)
        -> std::enable_if_t<std::is_same<To, bool>::value || !std::is_unsigned<To>::value || !std::is_floating_point<From>::value, To> {
    return static_cast<To>(from);
}

}  // namespace cuda
}  // namespace chainerx
