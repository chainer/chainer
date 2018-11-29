#pragma once

#include <cstdint>

namespace chainerx {
namespace native {

struct half {
public:
    half() {}
    half(const half& v) : data_(v.data_) {}
    explicit half(float v);

    explicit half(double v);
    explicit half(bool v);
    explicit half(int16_t v);
    explicit half(uint16_t v);
    explicit half(int32_t v);
    explicit half(uint32_t v);
    explicit half(int64_t v);
    explicit half(uint64_t v);

    operator float() const;
    explicit operator double() const;

private:
    uint16_t data_;
};
}  // namespace native
}  // namespace chainerx
