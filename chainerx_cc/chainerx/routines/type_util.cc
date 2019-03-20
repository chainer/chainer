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
        array_max_dtype_ = PromoteType(*array_max_dtype_, arg.dtype());
    } else {
        array_max_dtype_ = arg.dtype();
    }
}

void ResultTypeResolver::AddArg(Scalar arg) {
    if (scalar_max_dtype_.has_value()) {
        scalar_max_dtype_ = PromoteType(*scalar_max_dtype_, internal::GetDefaultDtype(arg.kind()));
    } else {
        scalar_max_dtype_ = internal::GetDefaultDtype(arg.kind());
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
        }
        return dt2;
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
    }
    // Otherwise return the signed dtype with one-level wider than the unsigned one.
    switch (dt2) {
        case Dtype::kUInt8:
            return Dtype::kInt16;
            // If there will be more unsigned int types, add here.
        default:
            CHAINERX_NEVER_REACH();
    }
}

}  // namespace type_util_detail
}  // namespace chainerx
