#pragma once

#include <cuda_fp16.hpp>

#include "chainerx/float16.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {

// Float16 for CUDA devices.
// Used from the device, it supports full arithmetics just like other C++ numerical types.
// Used from the host, it's only a 16-bit data storage.
class Float16 {
private:
    struct FromDataTag {};

public:
    __device__ Float16() : data_{0} {}
    explicit __device__ Float16(bool v) : Float16{static_cast<int>(v)} {}
    explicit __device__ Float16(int8_t v) : Float16{static_cast<int16_t>(v)} {}
    explicit __device__ Float16(uint8_t v) : Float16{static_cast<uint16_t>(v)} {}
    explicit __device__ Float16(int64_t v) : Float16{static_cast<int16_t>(v)} {}
    explicit __device__ Float16(uint64_t v) : Float16{static_cast<uint16_t>(v)} {}
    template <typename T>
    explicit __device__ Float16(T v) : Float16{::__half{v}} {}
    // It is assumed that chainerx::Float16 and cuda::Float16 have commmon representaiton.
    explicit __host__ Float16(Scalar v) : Float16{static_cast<chainerx::Float16>(v)} {}
    explicit __host__ Float16(chainerx::Float16 v) : Float16{v.data(), FromDataTag{}} {}

    explicit __device__ operator bool() const { return *this == Float16{0}; }
    // int8 conversion is not implemented in cuda_fp16
    explicit __device__ operator int8_t() const { return static_cast<int8_t>(static_cast<int16_t>(*this)); }
    explicit __device__ operator uint8_t() const { return static_cast<uint8_t>(static_cast<uint16_t>(*this)); }
    explicit __device__ operator int16_t() const { return static_cast<int16_t>(cuda_half()); }
    explicit __device__ operator uint16_t() const { return static_cast<uint16_t>(cuda_half()); }
    explicit __device__ operator int32_t() const { return static_cast<int32_t>(cuda_half()); }
    explicit __device__ operator uint32_t() const { return static_cast<uint32_t>(cuda_half()); }
    // int64 conversion is not implemented in cuda_fp16
    explicit __device__ operator int64_t() const { return static_cast<int32_t>(cuda_half()); }
    explicit __device__ operator uint64_t() const { return static_cast<uint32_t>(cuda_half()); }
    explicit __device__ operator float() const { return static_cast<float>(cuda_half()); }
    // double conversion is not implemented in cuda_fp16
    explicit __device__ operator double() const { return float{*this}; }

    // TODO(imanishi): Use cuda_half()
    __device__ Float16 operator-() const { return Float16{-static_cast<float>(*this)}; }
    __device__ bool operator!() const { return !static_cast<float>(*this); }
    __device__ Float16 operator+(const Float16& r) const { return Float16{static_cast<float>(*this) + static_cast<float>(r)}; }
    __device__ Float16 operator-(const Float16& r) const { return Float16{static_cast<float>(*this) - static_cast<float>(r)}; }
    __device__ Float16 operator*(const Float16& r) const { return Float16{static_cast<float>(*this) * static_cast<float>(r)}; }
    __device__ Float16 operator/(const Float16& r) const { return Float16{static_cast<float>(*this) / static_cast<float>(r)}; }
    __device__ Float16 operator+=(const Float16& r) { return *this = Float16{*this + r}; }
    __device__ Float16 operator-=(const Float16& r) { return *this = Float16{*this - r}; }
    __device__ Float16 operator*=(const Float16& r) { return *this = Float16{*this * r}; }
    __device__ Float16 operator/=(const Float16& r) { return *this = Float16{*this / r}; }
    __device__ bool operator==(const Float16& r) const { return static_cast<float>(*this) == static_cast<float>(r); }
    __device__ bool operator!=(const Float16& r) const { return !(*this == r); }
    __device__ bool operator<(const Float16& r) const { return static_cast<float>(*this) < static_cast<float>(r); }
    __device__ bool operator>(const Float16& r) const { return static_cast<float>(*this) > static_cast<float>(r); }
    __device__ bool operator<=(const Float16& r) const { return static_cast<float>(*this) <= static_cast<float>(r); }
    __device__ bool operator>=(const Float16& r) const { return static_cast<float>(*this) >= static_cast<float>(r); }

    __host__ __device__ static constexpr Float16 FromData(uint16_t data) { return cuda::Float16{data, FromDataTag{}}; }

    __host__ __device__ static constexpr Float16 Inf() { return FromData(0x7c00U); }
    __host__ __device__ static constexpr Float16 NegInf() { return FromData(0xfc00U); }

    __device__ bool IsNan() const { return (data_ & 0x7c00U) == 0x7c00U && (data_ & 0x03ffU) != 0; }
    __device__ bool IsInf() const { return (data_ & 0x7c00U) == 0x7c00U && (data_ & 0x03ffU) == 0; }
    __device__ Float16 Exp() const { return Float16{std::exp(static_cast<float>(*this))}; }
    __device__ Float16 Log() const { return Float16{std::log(static_cast<float>(*this))}; }
    __device__ Float16 Sqrt() const { return Float16{std::sqrt(static_cast<float>(*this))}; }
    __device__ Float16 Floor() const { return Float16{std::floor(static_cast<float>(*this))}; }

private:
    explicit __device__ Float16(::__half x) : data_{__half_as_ushort(x)} {}
    explicit __host__ __device__ constexpr Float16(uint16_t data, FromDataTag) : data_{data} {}

    __device__ ::__half cuda_half() const { return ::__half{__ushort_as_half(data_)}; }

    uint16_t data_;
};

template <typename T>
__device__ inline bool operator==(const T& l, const Float16& r) {
    return l == static_cast<float>(r);
}

template <typename T>
__device__ inline bool operator==(const Float16& l, const T& r) {
    return static_cast<float>(l) == r;
}

template <typename T>
__device__ inline bool operator!=(const T& l, const Float16& r) {
    return !(l == r);
}

template <typename T>
__device__ inline bool operator!=(const Float16& l, const T& r) {
    return !(l == r);
}

}  // namespace cuda
}  // namespace chainerx
