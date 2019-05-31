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
    // Suppress 'runtime/explicit' from cpplint, and 'google-explicit-constructor' and 'cppcoreguidelines-pro-type-member-init' from
    // clang-tidy.
    Scalar(bool v) : bool_{v}, kind_{DtypeKind::kBool} {}  // NOLINT
    Scalar(int8_t v) : int_{int64_t{v}}, kind_{DtypeKind::kInt} {}  // NOLINT
    Scalar(int16_t v) : int_{int64_t{v}}, kind_{DtypeKind::kInt} {}  // NOLINT
    Scalar(int32_t v) : int_{int64_t{v}}, kind_{DtypeKind::kInt} {}  // NOLINT
    Scalar(int64_t v) : int_{v}, kind_{DtypeKind::kInt} {}  // NOLINT
    Scalar(uint8_t v) : int_{int64_t{v}}, kind_{DtypeKind::kInt} {}  // NOLINT
    Scalar(uint16_t v) : int_{int64_t{v}}, kind_{DtypeKind::kInt} {}  // NOLINT
    Scalar(uint32_t v) : int_{int64_t{v}}, kind_{DtypeKind::kInt} {}  // NOLINT
    Scalar(Float16 v) : float_{static_cast<double>(v)}, kind_{DtypeKind::kFloat} {}  // NOLINT
    Scalar(float v) : float_{double{v}}, kind_{DtypeKind::kFloat} {}  // NOLINT
    Scalar(double v) : float_{v}, kind_{DtypeKind::kFloat} {}  // NOLINT

    template <typename T>
    Scalar(T v, DtypeKind kind) {
        switch (kind) {
            case DtypeKind::kBool:
                bool_ = static_cast<bool>(v);
                kind_ = DtypeKind::kBool;
                break;
            case DtypeKind::kInt:
            case DtypeKind::kUInt:
                int_ = static_cast<int64_t>(v);
                kind_ = DtypeKind::kInt;
                break;
            case DtypeKind::kFloat:
                float_ = static_cast<double>(v);
                kind_ = DtypeKind::kFloat;
                break;
            default:
                CHAINERX_NEVER_REACH();
        }
    }

    ~Scalar() = default;

    Scalar(const Scalar&) = default;
    Scalar(Scalar&&) = default;
    Scalar& operator=(const Scalar&) = default;
    Scalar& operator=(Scalar&&) = default;

    DtypeKind kind() const { return kind_; }

    std::string ToString() const;

    bool operator==(Scalar other) const {
        if (this->kind() == DtypeKind::kFloat || other.kind() == DtypeKind::kFloat) {
            return this->UnwrapAndCast<double>() == other.UnwrapAndCast<double>();
        }
        return this->UnwrapAndCast<int64_t>() == other.UnwrapAndCast<int64_t>();
    }

    bool operator!=(Scalar other) const { return !operator==(other); }

    Scalar& operator+() { return *this; }

    Scalar operator+() const { return *this; }

    Scalar operator-() const {
        switch (this->kind()) {
            case DtypeKind::kBool:
                throw DtypeError{"bool scalar cannot be negated"};
            case DtypeKind::kInt:
                return -int_;
            case DtypeKind::kFloat:
                return -float_;
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
        if (this->kind() == DtypeKind::kFloat || rhs.kind() == DtypeKind::kFloat) {
            return this->UnwrapAndCast<double>() + rhs.UnwrapAndCast<double>();
        }
        return this->UnwrapAndCast<int64_t>() + rhs.UnwrapAndCast<int64_t>();
    }

    Scalar operator-(const Scalar& rhs) const {
        if (this->kind() == DtypeKind::kFloat || rhs.kind() == DtypeKind::kFloat) {
            return this->UnwrapAndCast<double>() - rhs.UnwrapAndCast<double>();
        }
        return this->UnwrapAndCast<int64_t>() - rhs.UnwrapAndCast<int64_t>();
    }

    Scalar operator*(const Scalar& rhs) const {
        if (this->kind() == DtypeKind::kFloat || rhs.kind() == DtypeKind::kFloat) {
            return Scalar{this->UnwrapAndCast<double>() * rhs.UnwrapAndCast<double>()};
        }
        return Scalar{this->UnwrapAndCast<int64_t>() * rhs.UnwrapAndCast<int64_t>()};
    }

private:
    template <typename T>
    T UnwrapAndCast() const {
        switch (this->kind()) {
            case DtypeKind::kBool:
                return static_cast<T>(bool_);
            case DtypeKind::kInt:
                return static_cast<T>(int_);
            case DtypeKind::kFloat:
                return static_cast<T>(float_);
            default:
                CHAINERX_NEVER_REACH();
        }
        return T{};
    }

    union {
        bool bool_;
        int64_t int_;
        double float_;
    };

    DtypeKind kind_{};
};

std::ostream& operator<<(std::ostream& os, Scalar value);

}  // namespace chainerx
