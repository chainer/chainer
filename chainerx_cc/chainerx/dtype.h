#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include <gsl/gsl>

#include "chainerx/error.h"
#include "chainerx/float16.h"

namespace chainerx {

// NOTE: The dtype list is not fixed yet!!!
enum class Dtype {
    kBool = 1,
    kInt8,
    kInt16,
    kInt32,
    kInt64,
    kUInt8,
    kFloat16,
    kFloat32,
    kFloat64,
};

inline bool IsValidDtype(Dtype dtype) {
    using Underlying = std::underlying_type_t<Dtype>;
    auto value = static_cast<Underlying>(dtype);
    return 1 <= value && value <= static_cast<Underlying>(Dtype::kFloat64);
}

std::ostream& operator<<(std::ostream& os, Dtype dtype);

}  // namespace chainerx

namespace std {

template <>
struct hash<::chainerx::Dtype> {
    size_t operator()(::chainerx::Dtype dtype) const {
        using T = std::underlying_type_t<::chainerx::Dtype>;
        return hash<T>()(static_cast<T>(dtype));
    }
};

}  // namespace std

namespace chainerx {

// Kind of dtypes.
enum class DtypeKind {
    kBool = 0,
    kInt,
    kUInt,
    kFloat,
};

// Gets the single character identifier compatible to NumPy's dtype kind
inline char GetDtypeKindChar(DtypeKind kind) {
    switch (kind) {
        case DtypeKind::kBool:
            return 'b';
        case DtypeKind::kInt:
            return 'i';
        case DtypeKind::kUInt:
            return 'u';
        case DtypeKind::kFloat:
            return 'f';
        default:
            throw DtypeError{"invalid dtype kind"};
    }
}

template <typename T>
constexpr bool IsFloatingPointV = std::is_floating_point<T>::value || std::is_same<std::remove_const_t<T>, Float16>::value;

// Tag type used for dynamic dispatching with dtype value.
//
// This class template is used to resolve mapping from runtime dtype values to compile-time primitive types.
template <typename T>
struct PrimitiveType;

#define CHAINERX_DEFINE_PRIMITIVE_TYPE(name, code, dtype, kind, t, dst) \
    template <>                                                         \
    struct PrimitiveType<t> {                                           \
        using type = t;                                                 \
        using device_storage_type = dst;                                \
        static constexpr char kCharCode = code;                         \
        static constexpr Dtype kDtype = dtype;                          \
        static constexpr int64_t kElementSize = sizeof(type);           \
        static constexpr DtypeKind kKind = kind;                        \
        static const char* GetName() { return name; }                   \
    }

// TODO(niboshi): Char codes are mapped according to current development environment. They should be remapped depending on the executing
// environment, as in NumPy.
CHAINERX_DEFINE_PRIMITIVE_TYPE("bool", '?', Dtype::kBool, DtypeKind::kBool, bool, bool);
CHAINERX_DEFINE_PRIMITIVE_TYPE("int8", 'b', Dtype::kInt8, DtypeKind::kInt, int8_t, int8_t);
CHAINERX_DEFINE_PRIMITIVE_TYPE("int16", 'h', Dtype::kInt16, DtypeKind::kInt, int16_t, int16_t);
CHAINERX_DEFINE_PRIMITIVE_TYPE("int32", 'i', Dtype::kInt32, DtypeKind::kInt, int32_t, int32_t);
CHAINERX_DEFINE_PRIMITIVE_TYPE("int64", 'l', Dtype::kInt64, DtypeKind::kInt, int64_t, int64_t);
CHAINERX_DEFINE_PRIMITIVE_TYPE("uint8", 'B', Dtype::kUInt8, DtypeKind::kUInt, uint8_t, uint8_t);
CHAINERX_DEFINE_PRIMITIVE_TYPE("float16", 'e', Dtype::kFloat16, DtypeKind::kFloat, chainerx::Float16, uint16_t);
CHAINERX_DEFINE_PRIMITIVE_TYPE("float32", 'f', Dtype::kFloat32, DtypeKind::kFloat, float, float);
CHAINERX_DEFINE_PRIMITIVE_TYPE("float64", 'd', Dtype::kFloat64, DtypeKind::kFloat, double, double);

#undef CHAINERX_DEFINE_PRIMITIVE_TYPE

namespace dtype_detail {
template <typename To, typename From>
using WithConstnessOf = std::conditional_t<std::is_const<From>::value, std::add_const_t<To>, std::remove_const_t<To>>;
}

template <typename T>
using TypeToDeviceStorageType = dtype_detail::WithConstnessOf<typename PrimitiveType<std::remove_const_t<T>>::device_storage_type, T>;

// Dtype mapped from primitive type.
template <typename T>
constexpr Dtype TypeToDtype = PrimitiveType<std::remove_const_t<T>>::kDtype;

// Invokes a function by passing PrimitiveType<T> corresponding to given dtype value.
//
// For example,
//     VisitDtype(Dtype::kInt32, f, args...);
// is equivalent to
//     f(PrimtiveType<int>, args...);
// Note that the dtype argument can be a runtime value. This function can be used for dynamic dispatching based on dtype values.
//
// Note (beam2d): This function should be constexpr, but GCC 5.x does not allow it because of the throw statement, so currently not marked
// as constexpr.
template <typename F, typename... Args>
auto VisitDtype(Dtype dtype, F&& f, Args&&... args) {
    switch (dtype) {
        case Dtype::kBool:
            return std::forward<F>(f)(PrimitiveType<bool>{}, std::forward<Args>(args)...);
        case Dtype::kInt8:
            return std::forward<F>(f)(PrimitiveType<int8_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt16:
            return std::forward<F>(f)(PrimitiveType<int16_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt32:
            return std::forward<F>(f)(PrimitiveType<int32_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt64:
            return std::forward<F>(f)(PrimitiveType<int64_t>{}, std::forward<Args>(args)...);
        case Dtype::kUInt8:
            return std::forward<F>(f)(PrimitiveType<uint8_t>{}, std::forward<Args>(args)...);
        case Dtype::kFloat16:
            return std::forward<F>(f)(PrimitiveType<chainerx::Float16>{}, std::forward<Args>(args)...);
        case Dtype::kFloat32:
            return std::forward<F>(f)(PrimitiveType<float>{}, std::forward<Args>(args)...);
        case Dtype::kFloat64:
            return std::forward<F>(f)(PrimitiveType<double>{}, std::forward<Args>(args)...);
        default:
            throw DtypeError{"invalid dtype: ", static_cast<std::underlying_type_t<Dtype>>(dtype)};
    }
}

// Invokes a function by passing PrimitiveType<T> corresponding to given numeric dtype value.
// See VisitDtype for more detail.
template <typename F, typename... Args>
auto VisitNumericDtype(Dtype dtype, F&& f, Args&&... args) {
    switch (dtype) {
        case Dtype::kInt8:
            return std::forward<F>(f)(PrimitiveType<int8_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt16:
            return std::forward<F>(f)(PrimitiveType<int16_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt32:
            return std::forward<F>(f)(PrimitiveType<int32_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt64:
            return std::forward<F>(f)(PrimitiveType<int64_t>{}, std::forward<Args>(args)...);
        case Dtype::kUInt8:
            return std::forward<F>(f)(PrimitiveType<uint8_t>{}, std::forward<Args>(args)...);
        case Dtype::kFloat16:
            return std::forward<F>(f)(PrimitiveType<chainerx::Float16>{}, std::forward<Args>(args)...);
        case Dtype::kFloat32:
            return std::forward<F>(f)(PrimitiveType<float>{}, std::forward<Args>(args)...);
        case Dtype::kFloat64:
            return std::forward<F>(f)(PrimitiveType<double>{}, std::forward<Args>(args)...);
        default:
            throw DtypeError{"invalid dtype"};
    }
}

// Invokes a function by passing PrimitiveType<T> corresponding to given Integral dtype value.
// See VisitDtype for more detail.
template <typename F, typename... Args>
auto VisitIntegralDtype(Dtype dtype, F&& f, Args&&... args) {
    switch (dtype) {
        case Dtype::kBool:
            return std::forward<F>(f)(PrimitiveType<bool>{}, std::forward<Args>(args)...);
        case Dtype::kInt8:
            return std::forward<F>(f)(PrimitiveType<int8_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt16:
            return std::forward<F>(f)(PrimitiveType<int16_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt32:
            return std::forward<F>(f)(PrimitiveType<int32_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt64:
            return std::forward<F>(f)(PrimitiveType<int64_t>{}, std::forward<Args>(args)...);
        case Dtype::kUInt8:
            return std::forward<F>(f)(PrimitiveType<uint8_t>{}, std::forward<Args>(args)...);
        default:
            throw DtypeError{"invalid dtype"};
    }
}

// Invokes a function by passing PrimitiveType<T> corresponding to given Integral dtype value.
// See VisitDtype for more detail.
// TODO(sky58): Rename this to VisitIntegralDtype and current VisitIntegralDtype to something else.
template <typename F, typename... Args>
auto VisitShiftDtype(Dtype dtype, F&& f, Args&&... args) {
    switch (dtype) {
        case Dtype::kInt8:
            return std::forward<F>(f)(PrimitiveType<int8_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt16:
            return std::forward<F>(f)(PrimitiveType<int16_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt32:
            return std::forward<F>(f)(PrimitiveType<int32_t>{}, std::forward<Args>(args)...);
        case Dtype::kInt64:
            return std::forward<F>(f)(PrimitiveType<int64_t>{}, std::forward<Args>(args)...);
        case Dtype::kUInt8:
            return std::forward<F>(f)(PrimitiveType<uint8_t>{}, std::forward<Args>(args)...);
        default:
            throw DtypeError{"invalid dtype"};
    }
}

// Invokes a function by passing PrimitiveType<T> corresponding to given floating-point dtype value.
// See VisitDtype for more detail.
template <typename F, typename... Args>
auto VisitFloatingPointDtype(Dtype dtype, F&& f, Args&&... args) {
    switch (dtype) {
        case Dtype::kFloat16:
            return std::forward<F>(f)(PrimitiveType<chainerx::Float16>{}, std::forward<Args>(args)...);
        case Dtype::kFloat32:
            return std::forward<F>(f)(PrimitiveType<float>{}, std::forward<Args>(args)...);
        case Dtype::kFloat64:
            return std::forward<F>(f)(PrimitiveType<double>{}, std::forward<Args>(args)...);
        default:
            throw DtypeError{"invalid dtype"};
    }
}

// Gets the single character identifier compatible to NumPy's char code
inline char GetCharCode(Dtype dtype) {
    return VisitDtype(dtype, [](auto pt) { return decltype(pt)::kCharCode; });
}

// Gets the element size of the dtype in bytes.
inline int64_t GetItemSize(Dtype dtype) {
    return VisitDtype(dtype, [](auto pt) { return decltype(pt)::kElementSize; });
}

// Gets the kind of dtype.
inline DtypeKind GetKind(Dtype dtype) {
    return VisitDtype(dtype, [](auto pt) { return decltype(pt)::kKind; });
}

Dtype PromoteTypes(Dtype dt1, Dtype dt2);

// const char* representation of dtype compatible to NumPy's dtype name.
inline const char* GetDtypeName(Dtype dtype) {
    return VisitDtype(dtype, [](auto pt) { return decltype(pt)::GetName(); });
}

// Gets the dtype of given name.
Dtype GetDtype(const std::string& name);

// Returns a vector of all possible dtype values.
std::vector<Dtype> GetAllDtypes();

// Throws an exception if two dtypes mismatch.
void CheckEqual(Dtype lhs, Dtype rhs);

}  // namespace chainerx
