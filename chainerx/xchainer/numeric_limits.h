#pragma once

#include <cmath>
#include <cstdint>

#include "chainerx/macro.h"

namespace chainerx {

// Numerical limit values used in native and CUDA kernel implementations.
template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<bool> {
    XCHAINER_HOST_DEVICE static constexpr bool LowestOrInf() noexcept { return false; }
    XCHAINER_HOST_DEVICE static constexpr bool MaxOrInf() noexcept { return true; }
};

template <>
struct NumericLimits<uint8_t> {
    XCHAINER_HOST_DEVICE static constexpr uint8_t LowestOrInf() noexcept { return 0; }
    XCHAINER_HOST_DEVICE static constexpr uint8_t MaxOrInf() noexcept { return UINT8_MAX; }
};

template <>
struct NumericLimits<int8_t> {
    XCHAINER_HOST_DEVICE static constexpr int8_t LowestOrInf() noexcept { return INT8_MIN; }
    XCHAINER_HOST_DEVICE static constexpr int8_t MaxOrInf() noexcept { return INT8_MAX; }
};

template <>
struct NumericLimits<int16_t> {
    XCHAINER_HOST_DEVICE static constexpr int16_t LowestOrInf() noexcept { return INT16_MIN; }
    XCHAINER_HOST_DEVICE static constexpr int16_t MaxOrInf() noexcept { return INT16_MAX; }
};

template <>
struct NumericLimits<int32_t> {
    XCHAINER_HOST_DEVICE static constexpr int32_t LowestOrInf() noexcept { return INT32_MIN; }
    XCHAINER_HOST_DEVICE static constexpr int32_t MaxOrInf() noexcept { return INT32_MAX; }
};

template <>
struct NumericLimits<int64_t> {
    XCHAINER_HOST_DEVICE static constexpr int64_t LowestOrInf() noexcept { return INT64_MIN; }
    XCHAINER_HOST_DEVICE static constexpr int64_t MaxOrInf() noexcept { return INT64_MAX; }
};

template <>
struct NumericLimits<float> {
    XCHAINER_HOST_DEVICE static constexpr float LowestOrInf() noexcept { return -HUGE_VALF; }
    XCHAINER_HOST_DEVICE static constexpr float MaxOrInf() noexcept { return HUGE_VALF; }
};

template <>
struct NumericLimits<double> {
    XCHAINER_HOST_DEVICE static constexpr double LowestOrInf() noexcept { return -HUGE_VAL; }
    XCHAINER_HOST_DEVICE static constexpr double MaxOrInf() noexcept { return HUGE_VAL; }
};

}  // namespace chainerx
