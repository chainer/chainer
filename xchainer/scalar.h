#pragma once

#include <cassert>
#include <cstdint>
#include <ostream>

#include "xchainer/dtype.h"
#include "xchainer/error.h"

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

    std::string ToString() const;

    Scalar& operator+() { return *this; }

    Scalar operator+() const { return *this; }

    Scalar operator-() const {
        switch (dtype_) {
            case Dtype::kBool:
                throw DtypeError("bool scalar cannot be negated");
            case Dtype::kInt8:
                return -int8_;
            case Dtype::kInt16:
                return -int16_;
            case Dtype::kInt32:
                return -int32_;
            case Dtype::kInt64:
                return -int64_;
            case Dtype::kUInt8:
                // Negating unsigned
                return -uint8_;
            case Dtype::kFloat32:
                return -float32_;
            case Dtype::kFloat64:
                return -float64_;
            default:
                assert(0);  // should never be reached
        }
        return 0;
    }

    explicit operator bool() const { return UnwrapAndCast<bool>(); }
    explicit operator int8_t() const { return UnwrapAndCast<int8_t>(); }
    explicit operator int16_t() const { return UnwrapAndCast<int16_t>(); }
    explicit operator int32_t() const { return UnwrapAndCast<int32_t>(); }
    explicit operator int64_t() const { return UnwrapAndCast<int64_t>(); }
    explicit operator uint8_t() const { return UnwrapAndCast<uint8_t>(); }
    explicit operator float() const { return UnwrapAndCast<float>(); }
    explicit operator double() const { return UnwrapAndCast<double>(); }

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

    template <typename T>
    T UnwrapAndCast() const {
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
                assert(0);  // should never be reached
        }
    }
};

std::ostream& operator<<(std::ostream& os, Scalar value);

}  // namespace xchainer
