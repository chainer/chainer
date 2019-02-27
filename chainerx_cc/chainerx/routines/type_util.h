#pragma once

#include <utility>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/macro.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace type_util_detail {

class ResultTypeResolver {
public:
    template <typename Arg, typename... Args>
    Dtype ResolveArgs(Arg arg, Args... args) {
        // At least single argument is required.
        AddArgsImpl(std::forward<Arg>(arg));
        AddArgsImpl(std::forward<Args>(args)...);
        return Resolve();
    }

    Dtype Resolve() const {
        // If there were arrays, return the promoted array dtype.
        // Otherwise, return the promoted scalar dtype.
        if (array_max_dtype_.has_value()) {
            return *array_max_dtype_;
        }
        CHAINERX_ASSERT(scalar_max_dtype_.has_value());
        return *scalar_max_dtype_;
    }

    void AddArg(const Array& arg);

    void AddArg(Scalar arg);

private:
    nonstd::optional<Dtype> array_max_dtype_;
    nonstd::optional<Dtype> scalar_max_dtype_;

    // Returns the minimal dtype which can be safely casted from both dtypes.
    static Dtype PromoteType(Dtype dt1, Dtype dt2);

    void AddArgsImpl() {
        // nop
    }

    template <typename Arg, typename... Args>
    void AddArgsImpl(Arg arg, Args... args) {
        AddArg(std::forward<Arg>(arg));
        AddArgsImpl(std::forward<Args>(args)...);
    }
};

void ResultTypeResolver::AddArg(const Array& arg) {
    // If there already were arrays, compare with the promoted array dtype.
    // Othewise, keep the new dtype and forget scalars.
    if (array_max_dtype_.has_value()) {
        array_max_dtype_ = PromoteType(*array_max_dtype_, arg.dtype());
    } else {
        array_max_dtype_ = arg.dtype();
    }
    scalar_max_dtype_ = nonstd::nullopt;
}

void ResultTypeResolver::AddArg(Scalar arg) {
    // If there already were arrays, discard the scalar dtype.
    // Otherwise, compare with the promoted scalar dtype.
    if (array_max_dtype_.has_value()) {
        // discard the arg
    } else if (scalar_max_dtype_.has_value()) {
        scalar_max_dtype_ = PromoteType(*scalar_max_dtype_, arg.dtype());
    } else {
        scalar_max_dtype_ = arg.dtype();
    }
}

// Returns the minimal dtype which can be safely casted from both dtypes.
Dtype ResultTypeResolver::PromoteType(Dtype dt1, Dtype dt2) {
    DtypeKind kind1 = GetKind(dt1);
    DtypeKind kind2 = GetKind(dt2);
    // Bools always have least priority
    if (kind1 == DtypeKind::kBool) {
        return dt2;
    }
    if (kind2 == DtypeKind::kBool) {
        return dt1;
    }
    // Same kinds -> return the wider one
    if (kind1 == kind2) {
        if (GetItemSize(dt1) >= GetItemSize(dt2)) {
            return dt1;
        } else {
            return dt2;
        }
    }
    // Float takes priority over the other
    if (kind1 == DtypeKind::kFloat) {
        return dt1;
    }
    if (kind2 == DtypeKind::kFloat) {
        return dt2;
    }
    // Kinds are kInt and kUInt
    if (kind1 == DtypeKind::kUInt) {
        std::swap(dt1, dt2);
        std::swap(kind1, kind2);
    }
    CHAINERX_ASSERT(kind1 == DtypeKind::kInt && kind2 == DtypeKind::kUInt);
    if (GetItemSize(dt1) > GetItemSize(dt2)) {
        // Unsigned one has narrower width.
        // Return the signed dtype.
        return dt1;
    } else {
        // Otherwise return the signed dtype with one-level wider than the unsigned one.
        switch (dt2) {
            case Dtype::kUInt8:
                return Dtype::kInt16;
                // If there will be more unsigned int types, add here.
            default:
                CHAINERX_NEVER_REACH();
        }
    }
    CHAINERX_NEVER_REACH();
}

}  // namespace type_util_detail

inline Dtype ResultType(const Array& arg) { return arg.dtype(); }

inline Dtype ResultType(Scalar arg) { return arg.dtype(); }

template <typename Arg, typename... Args>
Dtype ResultType(Arg arg, Args... args) {
    return type_util_detail::ResultTypeResolver{}.ResolveArgs(std::forward<Arg>(arg), std::forward<Args>(args)...);
}

template <typename Container>
Dtype ResultType(Container args) {
    type_util_detail::ResultTypeResolver resolver{};
    if (args.size() == 0U) {
        throw ChainerxError{"At least one argument is required."};
    }
    for (const Array& arg : args) {
        resolver.AddArg(arg);
    }
    return resolver.Resolve();
}

}  // namespace chainerx
