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

// Kind of dtypes.
enum class DtypeKind {
    kBool = 0,
    kInt,
    kUInt,
    kFloat,
};

// Tag type used for dynamic dispatching with dtype value.
//
// This class template is used to resolve mapping from runtime dtype values to compile-time primitive types.
template <typename T>
struct PrimitiveType;

#define XCHAINER_DEFINE_PRIMITIVE_TYPE(name, code, dtype, kind, t) \
    template <>                                                    \
    struct PrimitiveType<t> {                                      \
        using type = t;                                            \
        static constexpr char kCharCode = code;                    \
        static constexpr Dtype kDtype = dtype;                     \
        static constexpr int64_t kElementSize = sizeof(type);      \
        static constexpr DtypeKind kKind = kind;                   \
        static const char* GetName() { return name; }              \
    }

XCHAINER_DEFINE_PRIMITIVE_TYPE("bool", '?', Dtype::kBool, DtypeKind::kBool, bool);
XCHAINER_DEFINE_PRIMITIVE_TYPE("int8", 'b', Dtype::kInt8, DtypeKind::kInt, int8_t);
XCHAINER_DEFINE_PRIMITIVE_TYPE("int16", 'h', Dtype::kInt16, DtypeKind::kInt, int16_t);
XCHAINER_DEFINE_PRIMITIVE_TYPE("int32", 'i', Dtype::kInt32, DtypeKind::kInt, int32_t);
XCHAINER_DEFINE_PRIMITIVE_TYPE("int64", 'q', Dtype::kInt64, DtypeKind::kInt, int64_t);
XCHAINER_DEFINE_PRIMITIVE_TYPE("uint8", 'B', Dtype::kUInt8, DtypeKind::kUInt, uint8_t);
XCHAINER_DEFINE_PRIMITIVE_TYPE("float32", 'f', Dtype::kFloat32, DtypeKind::kFloat, float);
XCHAINER_DEFINE_PRIMITIVE_TYPE("float64", 'd', Dtype::kFloat64, DtypeKind::kFloat, double);

#undef XCHAINER_DEFINE_PRIMITIVE_TYPE

// Dtype mapped from primitive type.
template <typename T>
constexpr Dtype TypeToDtype = PrimitiveType<T>::kDtype;

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
        case Dtype::kFloat32:
            return std::forward<F>(f)(PrimitiveType<float>{}, std::forward<Args>(args)...);
        case Dtype::kFloat64:
            return std::forward<F>(f)(PrimitiveType<double>{}, std::forward<Args>(args)...);
        default:
            throw DtypeError("invalid dtype");
    }
}

// Gets the single character identifier compatible to NumPy's char code
inline char GetCharCode(Dtype dtype) {
    return VisitDtype(dtype, [](auto pt) { return decltype(pt)::kCharCode; });
}

// Gets the element size of the dtype in bytes.
inline int64_t GetElementSize(Dtype dtype) {
    return VisitDtype(dtype, [](auto pt) { return decltype(pt)::kElementSize; });
}

// Gets the kind of dtype.
inline DtypeKind GetKind(Dtype dtype) {
    return VisitDtype(dtype, [](auto pt) { return decltype(pt)::kKind; });
}

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

}  // namespace xchainer
