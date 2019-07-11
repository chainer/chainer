#pragma once

#include <utility>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/macro.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace internal {

// Returns the default dtype.
inline Dtype GetDefaultDtype(DtypeKind kind) {
    switch (kind) {
        case DtypeKind::kBool:
            return Dtype::kBool;
        case DtypeKind::kInt:
            return Dtype::kInt32;
        case DtypeKind::kFloat:
            return Dtype::kFloat32;
        default:
            CHAINERX_NEVER_REACH();
    }
}

inline Dtype GetMathResultDtype(Dtype dtype) {
    if (GetKind(dtype) == DtypeKind::kFloat) {
        return dtype;
    }
    return Dtype::kFloat32;  // TODO(niboshi): Default dtype
}

}  // namespace internal

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

    Dtype Resolve() const;

    void AddArg(const Array& arg);

    void AddArg(Scalar arg);

private:
    absl::optional<Dtype> array_max_dtype_;
    absl::optional<Dtype> scalar_max_dtype_;

    void AddArgsImpl() {
        // nop
    }

    template <typename Arg, typename... Args>
    void AddArgsImpl(Arg arg, Args... args) {
        AddArg(std::forward<Arg>(arg));
        AddArgsImpl(std::forward<Args>(args)...);
    }

    static int GetDtypeCategory(Dtype dtype) {
        switch (GetKind(dtype)) {
            case DtypeKind::kFloat:
                return 2;
            default:
                return 1;
        }
    }
};

}  // namespace type_util_detail

inline Dtype ResultType(const Array& arg) { return arg.dtype(); }

inline Dtype ResultType(Scalar arg) { return internal::GetDefaultDtype(arg.kind()); }

template <typename Arg, typename... Args>
Dtype ResultType(Arg arg, Args... args) {
    return type_util_detail::ResultTypeResolver{}.ResolveArgs(std::forward<Arg>(arg), std::forward<Args>(args)...);
}

template <typename Container>
Dtype ResultType(Container args) {
    type_util_detail::ResultTypeResolver resolver{};
    if (args.empty()) {
        throw ChainerxError{"At least one argument is required."};
    }
    for (const Array& arg : args) {
        resolver.AddArg(arg);
    }
    return resolver.Resolve();
}

}  // namespace chainerx
