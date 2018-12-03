#pragma once

#include <cstdint>

namespace chainerx {
namespace native {

struct Half {
public:
    Half() {}
    Half(const Half& v) : data_(v.data_) {}
    explicit Half(float v);

    Half(double v);
    Half(bool v);
    Half(int16_t v);
    Half(uint16_t v);
    Half(int32_t v);
    Half(uint32_t v);
    Half(int64_t v);
    Half(uint64_t v);

    operator float() const;
    operator double() const;

    friend bool operator==(const Half& lhs, const Half& rhs);
    Half operator-() const;
    Half operator+(const Half& r) const;
    Half operator-(const Half& r) const;
    Half operator*(const Half& r) const;
    Half operator/(const Half& r) const;
    Half& operator+=(const Half& r);
    Half& operator-=(const Half& r);
    Half& operator*=(const Half& r);
    Half& operator/=(const Half& r);

private:
    uint16_t data_;
};

}  // namespace native
}  // namespace chainerx
