#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include <gsl/gsl>

#include "xchainer/error.h"

namespace xchainer {

// NOTE: The dtype list is not fixed yet!!!
enum class Dtype {
    kBool = 1,
    kInt8,
    kInt16,
    kInt32,
    kInt64,
    kUInt8,
    kFloat32,
    kFloat64,
};

inline bool IsValidDtype(Dtype dtype) {
    using Underlying = std::underlying_type_t<Dtype>;
    auto value = static_cast<Underlying>(dtype);
    return 1 <= value && value <= static_cast<Underlying>(Dtype::kFloat64);
}

std::ostream& operator<<(std::ostream& os, Dtype dtype);

}  // namespace xchainer

namespace std {

template <>
struct hash<::xchainer::Dtype> {
    size_t operator()(::xchainer::Dtype dtype) const {
        using T = std::underlying_type_t<::xchainer::Dtype>;
        return hash<T>()(static_cast<T>(dtype));
    }
};

}  // namespace std

namespace xchainer {

namespace dtype_detail {

template <typename T>
struct TypeToDtype;
template <>
struct TypeToDtype<bool> {
    static constexpr Dtype value = Dtype::kBool;
};
template <>
struct TypeToDtype<int8_t> {
    static constexpr Dtype value = Dtype::kInt8;
};
template <>
struct TypeToDtype<int16_t> {
    static constexpr Dtype value = Dtype::kInt16;
};
template <>
struct TypeToDtype<int32_t> {
    static constexpr Dtype value = Dtype::kInt32;
};
template <>
struct TypeToDtype<int64_t> {
    static constexpr Dtype value = Dtype::kInt64;
};
template <>
struct TypeToDtype<uint8_t> {
    static constexpr Dtype value = Dtype::kUInt8;
};
template <>
struct TypeToDtype<float> {
    static constexpr Dtype value = Dtype::kFloat32;
};
template <>
struct TypeToDtype<double> {
    static constexpr Dtype value = Dtype::kFloat64;
};

template <char c>
struct CharToDtype;
template <>
struct CharToDtype<'?'> {
    static constexpr Dtype value = Dtype::kBool;
};
template <>
struct CharToDtype<'b'> {
    static constexpr Dtype value = Dtype::kInt8;
};
template <>
struct CharToDtype<'h'> {
    static constexpr Dtype value = Dtype::kInt16;
};
template <>
struct CharToDtype<'i'> {
    static constexpr Dtype value = Dtype::kInt32;
};
template <>
struct CharToDtype<'q'> {
    static constexpr Dtype value = Dtype::kInt64;
};
template <>
struct CharToDtype<'B'> {
    static constexpr Dtype value = Dtype::kUInt8;
};
template <>
struct CharToDtype<'f'> {
    static constexpr Dtype value = Dtype::kFloat32;
};
template <>
struct CharToDtype<'d'> {
    static constexpr Dtype value = Dtype::kFloat64;
};

template <Dtype dtype>
struct DtypeToType;
template <>
struct DtypeToType<Dtype::kBool> {
    using type = bool;
};
template <>
struct DtypeToType<Dtype::kInt8> {
    using type = int8_t;
};
template <>
struct DtypeToType<Dtype::kInt16> {
    using type = int16_t;
};
template <>
struct DtypeToType<Dtype::kInt32> {
    using type = int32_t;
};
template <>
struct DtypeToType<Dtype::kInt64> {
    using type = int64_t;
};
template <>
struct DtypeToType<Dtype::kUInt8> {
    using type = uint8_t;
};
template <>
struct DtypeToType<Dtype::kFloat32> {
    using type = float;
};
template <>
struct DtypeToType<Dtype::kFloat64> {
    using type = double;
};

}  // namespace dtype_detail

// TypeToDtype<type> == dtype
template <typename T>
constexpr Dtype TypeToDtype = dtype_detail::TypeToDtype<T>::value;

// CharToDtype<c> == dtype
template <char c>
constexpr Dtype CharToDtype = dtype_detail::CharToDtype<c>::value;

// DtypeToType<dtype> == type
template <Dtype dtype>
using DtypeToType = typename dtype_detail::DtypeToType<dtype>::type;

// Single character identifier compatible to NumPy's char code
constexpr char GetCharCode(Dtype dtype) {
    switch (dtype) {
        case Dtype::kBool:
            return '?';
        case Dtype::kInt8:
            return 'b';
        case Dtype::kInt16:
            return 'h';
        case Dtype::kInt32:
            return 'i';
        case Dtype::kInt64:
            return 'q';
        case Dtype::kUInt8:
            return 'B';
        case Dtype::kFloat32:
            return 'f';
        case Dtype::kFloat64:
            return 'd';
    }
    return 0;  // never happen
}

constexpr int64_t GetElementSize(Dtype dtype) {
    switch (dtype) {
        case Dtype::kBool:
            return 1;
        case Dtype::kInt8:
            return 1;
        case Dtype::kInt16:
            return 2;
        case Dtype::kInt32:
            return 4;
        case Dtype::kInt64:
            return 8;
        case Dtype::kUInt8:
            return 1;
        case Dtype::kFloat32:
            return 4;
        case Dtype::kFloat64:
            return 8;
    }
    return 0;  // never happen
}

// const char* representation of dtype compatible to NumPy's dtype name.
const char* GetDtypeName(Dtype dtype);

// Gets the dtype of given name.
Dtype GetDtype(const std::string& name);

// Returns a vector of all possible dtype values.
std::vector<Dtype> GetAllDtypes();

// Throws an exception if two dtypes mismatch.
void CheckEqual(Dtype lhs, Dtype rhs);

// Kind of dtypes.
enum class DtypeKind {
    kBool = 0,
    kInt,
    kUInt,
    kFloat,
};

constexpr DtypeKind GetKind(Dtype dtype) {
    switch (dtype) {
        case Dtype::kBool:
            return DtypeKind::kBool;
        case Dtype::kInt8:
        case Dtype::kInt16:
        case Dtype::kInt32:
        case Dtype::kInt64:
            return DtypeKind::kInt;
        case Dtype::kUInt8:
            return DtypeKind::kUInt;
        case Dtype::kFloat32:
        case Dtype::kFloat64:
            return DtypeKind::kFloat;
    }
    return DtypeKind::kInt;  // never happen
}

}  // namespace xchainer
