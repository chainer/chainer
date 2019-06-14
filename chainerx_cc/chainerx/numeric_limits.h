#pragma once

#include <cmath>
#include <cstdint>

#ifdef _WIN32
#include <limits>
#endif  // _WIN32

#include "chainerx/float16.h"
#include "chainerx/macro.h"

namespace chainerx {

// Numerical limit values used in native and CUDA kernel implementations.
template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<bool> {
    CHAINERX_HOST_DEVICE static constexpr bool LowestOrInf() noexcept { return false; }
    CHAINERX_HOST_DEVICE static constexpr bool MaxOrInf() noexcept { return true; }
};

template <>
struct NumericLimits<uint8_t> {
    CHAINERX_HOST_DEVICE static constexpr uint8_t LowestOrInf() noexcept { return 0; }
    CHAINERX_HOST_DEVICE static constexpr uint8_t MaxOrInf() noexcept { return UINT8_MAX; }
};

template <>
struct NumericLimits<int8_t> {
    CHAINERX_HOST_DEVICE static constexpr int8_t LowestOrInf() noexcept { return INT8_MIN; }
    CHAINERX_HOST_DEVICE static constexpr int8_t MaxOrInf() noexcept { return INT8_MAX; }
};

template <>
struct NumericLimits<int16_t> {
    CHAINERX_HOST_DEVICE static constexpr int16_t LowestOrInf() noexcept { return INT16_MIN; }
    CHAINERX_HOST_DEVICE static constexpr int16_t MaxOrInf() noexcept { return INT16_MAX; }
};

template <>
struct NumericLimits<int32_t> {
    CHAINERX_HOST_DEVICE static constexpr int32_t LowestOrInf() noexcept { return INT32_MIN; }
    CHAINERX_HOST_DEVICE static constexpr int32_t MaxOrInf() noexcept { return INT32_MAX; }
};

template <>
struct NumericLimits<int64_t> {
    CHAINERX_HOST_DEVICE static constexpr int64_t LowestOrInf() noexcept { return INT64_MIN; }
    CHAINERX_HOST_DEVICE static constexpr int64_t MaxOrInf() noexcept { return INT64_MAX; }
};

template <>
struct NumericLimits<chainerx::Float16> {
    CHAINERX_HOST_DEVICE static constexpr chainerx::Float16 LowestOrInf() noexcept { return chainerx::Float16::FromData(0xfc00); }
    CHAINERX_HOST_DEVICE static constexpr chainerx::Float16 MaxOrInf() noexcept { return chainerx::Float16::FromData(0x7c00); }
};

#ifdef _WIN32
template <>
struct NumericLimits<float> {
    CHAINERX_HOST_DEVICE static constexpr float LowestOrInf() noexcept { return -std::numeric_limits<float>::infinity(); }
    CHAINERX_HOST_DEVICE static constexpr float MaxOrInf() noexcept { return std::numeric_limits<float>::infinity(); }
};

template <>
struct NumericLimits<double> {
    CHAINERX_HOST_DEVICE static constexpr double LowestOrInf() noexcept { return -std::numeric_limits<double>::infinity(); }
    CHAINERX_HOST_DEVICE static constexpr double MaxOrInf() noexcept { return std::numeric_limits<double>::infinity(); }
};
#else  // _WIN32
template <>
struct NumericLimits<float> {
    CHAINERX_HOST_DEVICE static constexpr float LowestOrInf() noexcept { return -HUGE_VALF; }
    CHAINERX_HOST_DEVICE static constexpr float MaxOrInf() noexcept { return HUGE_VALF; }
};

template <>
struct NumericLimits<double> {
    CHAINERX_HOST_DEVICE static constexpr double LowestOrInf() noexcept { return -HUGE_VAL; }
    CHAINERX_HOST_DEVICE static constexpr double MaxOrInf() noexcept { return HUGE_VAL; }
};
#endif  // _WIN32

}  // namespace chainerx
