#pragma once

#include <cstdint>
#include <ostream>
#include <string>

#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/float16.h"
#include "chainerx/macro.h"

namespace chainerx {

// Type safe, dynamically typed scalar value.
class Scalar {
public:
    Scalar(bool v) : bool_{v}, dtype_{Dtype::kBool} {}  // NOLINT(runtime/explicit)
    Scalar(int8_t v) : int8_{v}, dtype_{Dtype::kInt8} {}  // NOLINT(runtime/explicit)
    Scalar(int16_t v) : int16_{v}, dtype_{Dtype::kInt16} {}  // NOLINT(runtime/explicit)
    Scalar(int32_t v) : int32_{v}, dtype_{Dtype::kInt32} {}  // NOLINT(runtime/explicit)
    Scalar(int64_t v) : int64_{v}, dtype_{Dtype::kInt64} {}  // NOLINT(runtime/explicit)
    Scalar(uint8_t v) : uint8_{v}, dtype_{Dtype::kUInt8} {}  // NOLINT(runtime/explicit)
    Scalar(Float16 v) : float16_{v}, dtype_{Dtype::kFloat16} {}  // NOLINT(runtime/explicit)
    Scalar(float v) : float32_{v}, dtype_{Dtype::kFloat32} {}  // NOLINT(runtime/explicit)
    Scalar(double v) : float64_{v}, dtype_{Dtype::kFloat64} {}  // NOLINT(runtime/explicit)

    template <typename T>
    Scalar(T v, Dtype dtype) : dtype_{dtype} {
        switch (dtype) {
            case Dtype::kBool:
                bool_ = static_cast<bool>(v);
                break;
            case Dtype::kInt8:
                int8_ = static_cast<int8_t>(v);
                break;
            case Dtype::kInt16:
                int16_ = static_cast<int16_t>(v);
                break;
            case Dtype::kInt32:
                int32_ = static_cast<int32_t>(v);
                break;
            case Dtype::kInt64:
                int64_ = static_cast<int64_t>(v);
                break;
            case Dtype::kUInt8:
                uint8_ = static_cast<uint8_t>(v);
                break;
            case Dtype::kFloat16:
                float16_ = static_cast<Float16>(v);
                break;
            case Dtype::kFloat32:
                float32_ = static_cast<float>(v);
                break;
            case Dtype::kFloat64:
                float64_ = static_cast<double>(v);
                break;
            default:
                CHAINERX_NEVER_REACH();
        }
    }

    Scalar(const Scalar&) = default;
    Scalar& operator=(const Scalar&) = default;

    Dtype dtype() const { return dtype_; }

    std::string ToString() const;

    bool operator==(Scalar other) const {
        // TODO(niboshi): Support different dtypes
        if (dtype_ != other.dtype_) {
            return false;
        }

        return VisitDtype(dtype_, [this, other](auto pt) {
            using T = typename decltype(pt)::type;
            return this->UnwrapAndCast<T>() == other.UnwrapAndCast<T>();
        });
    }

    bool operator!=(Scalar other) const { return !operator==(other); }

    Scalar& operator+() { return *this; }

    Scalar operator+() const { return *this; }

    Scalar operator-() const {
        switch (dtype_) {
            case Dtype::kBool:
                throw DtypeError{"bool scalar cannot be negated"};
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
            case Dtype::kFloat16:
                return -float16_;
            case Dtype::kFloat32:
                return -float32_;
            case Dtype::kFloat64:
                return -float64_;
            default:
                CHAINERX_NEVER_REACH();
        }
        return 0;
    }

    explicit operator bool() const { return UnwrapAndCast<bool>(); }
    explicit operator int8_t() const { return UnwrapAndCast<int8_t>(); }
    explicit operator int16_t() const { return UnwrapAndCast<int16_t>(); }
    explicit operator int32_t() const { return UnwrapAndCast<int32_t>(); }
    explicit operator int64_t() const { return UnwrapAndCast<int64_t>(); }
    explicit operator uint8_t() const { return UnwrapAndCast<uint8_t>(); }
    explicit operator Float16() const { return UnwrapAndCast<Float16>(); }
    explicit operator float() const { return UnwrapAndCast<float>(); }
    explicit operator double() const { return UnwrapAndCast<double>(); }

    Scalar operator+(const Scalar& rhs) const {
        // TODO(niboshi): Support dtype conversion
        CHAINERX_ASSERT(dtype_ == rhs.dtype_);

        return VisitDtype(dtype_, [&](auto pt) {
            using T = typename decltype(pt)::type;
            return Scalar{static_cast<T>(*this) + static_cast<T>(rhs)};
        });
    }

    Scalar operator*(const Scalar& rhs) const {
        // TODO(niboshi): Support dtype conversion
        CHAINERX_ASSERT(dtype_ == rhs.dtype_);

        return VisitDtype(dtype_, [&](auto pt) {
            using T = typename decltype(pt)::type;
            return Scalar{static_cast<T>(*this) * static_cast<T>(rhs)};
        });
    }

private:
    template <typename T>
    T UnwrapAndCast() const {
        switch (dtype_) {
            case Dtype::kBool:
                return static_cast<T>(bool_);
            case Dtype::kInt8:
                return static_cast<T>(int8_);
            case Dtype::kInt16:
                return static_cast<T>(int16_);
            case Dtype::kInt32:
                return static_cast<T>(int32_);
            case Dtype::kInt64:
                return static_cast<T>(int64_);
            case Dtype::kUInt8:
                return static_cast<T>(uint8_);
            case Dtype::kFloat16:
                return static_cast<T>(float16_);
            case Dtype::kFloat32:
                return static_cast<T>(float32_);
            case Dtype::kFloat64:
                return static_cast<T>(float64_);
            default:
                CHAINERX_NEVER_REACH();
        }
        return T{};
    }

    union {
        bool bool_;
        int8_t int8_;
        int16_t int16_;
        int32_t int32_;
        int64_t int64_;
        uint8_t uint8_;
        Float16 float16_;
        float float32_;
        double float64_;
    };

    Dtype dtype_;
};

std::ostream& operator<<(std::ostream& os, Scalar value);

}  // namespace chainerx
