#pragma once

#include <cstdint>

namespace chainerx {

class Half {
private:
    struct FromDataTag {};

public:
    Half() {}
    explicit Half(float v);
    explicit Half(double v);

    explicit Half(bool v) : Half{static_cast<float>(v)} {}
    explicit Half(int16_t v) : Half{static_cast<float>(v)} {}
    explicit Half(uint16_t v) : Half{static_cast<float>(v)} {}
    explicit Half(int32_t v) : Half{static_cast<double>(v)} {}
    explicit Half(uint32_t v) : Half{static_cast<double>(v)} {}
    explicit Half(int64_t v) : Half{static_cast<double>(v)} {}
    explicit Half(uint64_t v) : Half{static_cast<double>(v)} {}

    explicit operator float() const;
    explicit operator double() const;

    explicit operator bool() const { return static_cast<float>(*this); }
    explicit operator int16_t() const { return static_cast<float>(*this); }
    explicit operator uint16_t() const { return static_cast<float>(*this); }
    explicit operator int32_t() const { return static_cast<double>(*this); }
    explicit operator uint32_t() const { return static_cast<double>(*this); }
    explicit operator int64_t() const { return static_cast<double>(*this); }
    explicit operator uint64_t() const { return static_cast<double>(*this); }
    explicit operator signed char() const { return static_cast<float>(*this); }
    explicit operator unsigned char() const { return static_cast<float>(*this); }

    bool operator==(const Half& r) const { return static_cast<float>(*this) == static_cast<float>(r); }
    bool operator!=(const Half& r) const { return static_cast<float>(*this) != static_cast<float>(r); }
    bool operator<(const Half& r) const { return static_cast<float>(*this) < static_cast<float>(r); }
    bool operator>(const Half& r) const { return static_cast<float>(*this) > static_cast<float>(r); }
    bool operator<=(const Half& r) const { return static_cast<float>(*this) <= static_cast<float>(r); }
    bool operator>=(const Half& r) const { return static_cast<float>(*this) >= static_cast<float>(r); }
    Half operator-() const { return Half{-static_cast<float>(*this)}; }
    Half operator+(const Half& r) const { return Half{static_cast<float>(*this) + static_cast<float>(r)}; }
    Half operator-(const Half& r) const { return Half{static_cast<float>(*this) - static_cast<float>(r)}; }
    Half operator*(const Half& r) const { return Half{static_cast<float>(*this) * static_cast<float>(r)}; }
    Half operator/(const Half& r) const { return Half{static_cast<float>(*this) / static_cast<float>(r)}; }
    Half& operator+=(const Half& r) { return *this = *this + r; }
    Half& operator-=(const Half& r) { return *this = *this - r; }
    Half& operator*=(const Half& r) { return *this = *this * r; }
    Half& operator/=(const Half& r) { return *this = *this / r; }

    uint16_t data() const { return data_; }
    static Half FromData(uint16_t data) { return Half{data, FromDataTag{}}; }

private:
    explicit Half(uint16_t data, FromDataTag) : data_{data} {}
    uint16_t data_;
};
}  // namespace chainerx
