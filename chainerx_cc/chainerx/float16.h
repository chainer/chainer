#pragma once

#include <cstdint>

#include "chainerx/macro.h"

namespace chainerx {

class Float16 {
private:
    struct FromDataTag {};

public:
    // NOLINT is required since `= default` does now work with CUDA.
    // NOLINTNEXTLINE(modernize-use-equals-default)
    CHAINERX_HOST_DEVICE Float16() {}
    CHAINERX_HOST_DEVICE explicit Float16(float v);
    CHAINERX_HOST_DEVICE explicit Float16(double v);

    CHAINERX_HOST_DEVICE explicit Float16(bool v) : Float16{static_cast<float>(v)} {}
    CHAINERX_HOST_DEVICE explicit Float16(int16_t v) : Float16{static_cast<float>(v)} {}
    CHAINERX_HOST_DEVICE explicit Float16(uint16_t v) : Float16{static_cast<float>(v)} {}
    CHAINERX_HOST_DEVICE explicit Float16(int32_t v) : Float16{static_cast<double>(v)} {}
    CHAINERX_HOST_DEVICE explicit Float16(uint32_t v) : Float16{static_cast<double>(v)} {}
    CHAINERX_HOST_DEVICE explicit Float16(int64_t v) : Float16{static_cast<double>(v)} {}
    CHAINERX_HOST_DEVICE explicit Float16(uint64_t v) : Float16{static_cast<double>(v)} {}

    CHAINERX_HOST_DEVICE explicit operator float() const;
    CHAINERX_HOST_DEVICE explicit operator double() const;

    CHAINERX_HOST_DEVICE explicit operator bool() const { return static_cast<float>(*this) != 0.0f; }
    CHAINERX_HOST_DEVICE explicit operator int16_t() const { return static_cast<int16_t>(static_cast<float>(*this)); }
    CHAINERX_HOST_DEVICE explicit operator uint16_t() const { return static_cast<uint16_t>(static_cast<float>(*this)); }
    CHAINERX_HOST_DEVICE explicit operator int32_t() const { return static_cast<int32_t>(static_cast<double>(*this)); }
    CHAINERX_HOST_DEVICE explicit operator uint32_t() const { return static_cast<uint32_t>(static_cast<double>(*this)); }
    CHAINERX_HOST_DEVICE explicit operator int64_t() const { return static_cast<int64_t>(static_cast<double>(*this)); }
    CHAINERX_HOST_DEVICE explicit operator uint64_t() const { return static_cast<uint64_t>(static_cast<double>(*this)); }
    CHAINERX_HOST_DEVICE explicit operator int8_t() const { return static_cast<int8_t>(static_cast<float>(*this)); }
    CHAINERX_HOST_DEVICE explicit operator uint8_t() const { return static_cast<uint8_t>(static_cast<float>(*this)); }

    CHAINERX_HOST_DEVICE bool operator==(Float16 r) const { return static_cast<float>(*this) == static_cast<float>(r); }
    CHAINERX_HOST_DEVICE bool operator!=(Float16 r) const { return static_cast<float>(*this) != static_cast<float>(r); }
    CHAINERX_HOST_DEVICE bool operator<(Float16 r) const { return static_cast<float>(*this) < static_cast<float>(r); }
    CHAINERX_HOST_DEVICE bool operator>(Float16 r) const { return static_cast<float>(*this) > static_cast<float>(r); }
    CHAINERX_HOST_DEVICE bool operator<=(Float16 r) const { return static_cast<float>(*this) <= static_cast<float>(r); }
    CHAINERX_HOST_DEVICE bool operator>=(Float16 r) const { return static_cast<float>(*this) >= static_cast<float>(r); }
    CHAINERX_HOST_DEVICE Float16 operator-() const { return Float16{-static_cast<float>(*this)}; }
    CHAINERX_HOST_DEVICE Float16 operator+(Float16 r) const { return Float16{static_cast<float>(*this) + static_cast<float>(r)}; }
    CHAINERX_HOST_DEVICE Float16 operator-(Float16 r) const { return Float16{static_cast<float>(*this) - static_cast<float>(r)}; }
    CHAINERX_HOST_DEVICE Float16 operator*(Float16 r) const { return Float16{static_cast<float>(*this) * static_cast<float>(r)}; }
    CHAINERX_HOST_DEVICE Float16 operator/(Float16 r) const { return Float16{static_cast<float>(*this) / static_cast<float>(r)}; }
    CHAINERX_HOST_DEVICE Float16& operator+=(Float16 r) { return *this = *this + r; }
    CHAINERX_HOST_DEVICE Float16& operator-=(Float16 r) { return *this = *this - r; }
    CHAINERX_HOST_DEVICE Float16& operator*=(Float16 r) { return *this = *this * r; }
    CHAINERX_HOST_DEVICE Float16& operator/=(Float16 r) { return *this = *this / r; }

    CHAINERX_HOST_DEVICE uint16_t data() const { return data_; }
    CHAINERX_HOST_DEVICE static constexpr Float16 FromData(uint16_t data) { return Float16{data, FromDataTag{}}; }

    bool IsNan() const { return (data_ & 0x7c00U) == 0x7c00U && (data_ & 0x03ffU) != 0; }
    bool IsInf() const { return (data_ & 0x7c00U) == 0x7c00U && (data_ & 0x03ffU) == 0; }

private:
    CHAINERX_HOST_DEVICE constexpr Float16(uint16_t data, FromDataTag /*tag*/) : data_{data} {}
    uint16_t data_{};
};

template <typename T>
CHAINERX_HOST_DEVICE inline bool operator==(const T& l, Float16 r) {
    return l == static_cast<float>(r);
}

template <typename T>
CHAINERX_HOST_DEVICE inline bool operator==(Float16 l, const T& r) {
    return static_cast<float>(l) == r;
}

template <typename T>
CHAINERX_HOST_DEVICE inline bool operator!=(const T& l, Float16 r) {
    return l != static_cast<float>(r);
}

template <typename T>
CHAINERX_HOST_DEVICE inline bool operator!=(Float16 l, const T& r) {
    return static_cast<float>(l) != r;
}

}  // namespace chainerx
