#include "chainerx/routines/type_util.h"

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/macro.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace type_util_detail {

Dtype ResultTypeResolver::Resolve() const {
    // If there were arrays, return the promoted array dtype.
    // Otherwise, return the promoted scalar dtype.
    if (array_max_dtype_.has_value()) {
        Dtype array_max_dtype = *array_max_dtype_;
        if (scalar_max_dtype_.has_value()) {
            Dtype scalar_max_dtype = *scalar_max_dtype_;
            if (GetDtypeCategory(scalar_max_dtype) > GetDtypeCategory(array_max_dtype)) {
                return scalar_max_dtype;
            }
        }
        return array_max_dtype;
    }
    CHAINERX_ASSERT(scalar_max_dtype_.has_value());
    return *scalar_max_dtype_;
}

void ResultTypeResolver::AddArg(const Array& arg) {
    // If there already were arrays, compare with the promoted array dtype.
    // Othewise, keep the new dtype and forget scalars.
    if (array_max_dtype_.has_value()) {
        array_max_dtype_ = PromoteTypes(*array_max_dtype_, arg.dtype());
    } else {
        array_max_dtype_ = arg.dtype();
    }
}

void ResultTypeResolver::AddArg(Scalar arg) {
    if (scalar_max_dtype_.has_value()) {
        scalar_max_dtype_ = PromoteTypes(*scalar_max_dtype_, internal::GetDefaultDtype(arg.kind()));
    } else {
        scalar_max_dtype_ = internal::GetDefaultDtype(arg.kind());
    }
}

}  // namespace type_util_detail
}  // namespace chainerx
