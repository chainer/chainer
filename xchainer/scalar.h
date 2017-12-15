#pragma once

#include <cstdint>
#include <ostream>

#include "xchainer/dtype.h"

namespace xchainer {

// Type safe, dynamically typed scalar value.
class Scalar {
public:
    Scalar(bool v) : bool_(v), dtype_(Dtype::kBool) {}
    Scalar(int8_t v) : int8_(v), dtype_(Dtype::kInt8) {}
    Scalar(int16_t v) : int16_(v), dtype_(Dtype::kInt16) {}
    Scalar(int32_t v) : int32_(v), dtype_(Dtype::kInt32) {}
    Scalar(int64_t v) : int64_(v), dtype_(Dtype::kInt64) {}
    Scalar(uint8_t v) : uint8_(v), dtype_(Dtype::kUInt8) {}
    Scalar(float v) : float32_(v), dtype_(Dtype::kFloat32) {}
    Scalar(double v) : float64_(v), dtype_(Dtype::kFloat64) {}

    Scalar(const Scalar&) = default;
    Scalar& operator=(const Scalar&) = default;

    Dtype dtype() const { return dtype_; }

    template <typename T>
    T Cast() const {
        switch (dtype_) {
            case Dtype::kBool:
                return bool_;
            case Dtype::kInt8:
                return int8_;
            case Dtype::kInt16:
                return int16_;
            case Dtype::kInt32:
                return int32_;
            case Dtype::kInt64:
                return int64_;
            case Dtype::kUInt8:
                return uint8_;
            case Dtype::kFloat32:
                return float32_;
            case Dtype::kFloat64:
                return float64_;
            default:
                return 0;  // never reach
        }
    }

    std::string ToString() const;

    Scalar& operator+() { return *this; }

    Scalar operator+() const { return *this; }

    operator bool() const { return Cast<bool>(); }
    operator int8_t() const { return Cast<int8_t>(); }
    operator int16_t() const { return Cast<int16_t>(); }
    operator int32_t() const { return Cast<int32_t>(); }
    operator int64_t() const { return Cast<int64_t>(); }
    operator uint8_t() const { return Cast<uint8_t>(); }
    operator float() const { return Cast<float>(); }
    operator double() const { return Cast<double>(); }

private:
    union {
        bool bool_;
        int8_t int8_;
        int16_t int16_;
        int32_t int32_;
        int64_t int64_;
        uint8_t uint8_;
        float float32_;
        double float64_;
    };
    Dtype dtype_;
};

Scalar operator-(Scalar value);

std::ostream& operator<<(std::ostream& os, Scalar value);

}  // namespace xchainer
