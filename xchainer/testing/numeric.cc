#include "xchainer/testing/numeric.h"

#include <cassert>

#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace testing {

template <typename T>
bool AllCloseImpl(const Array& a, const Array& b, const Scalar& atol, const Scalar& rtol) {
    auto total_size = a.shape().total_size();
    const T* adata = static_cast<const T*>(a.data().get());
    const T* bdata = static_cast<const T*>(b.data().get());
    const T at = static_cast<const T>(atol);
    const T rt = static_cast<const T>(rtol);
    for (decltype(total_size) i = 0; i < total_size; i++) {
        if (std::abs(adata[i] - bdata[i]) > (at + rt * std::abs(bdata[i]))) {
            return false;
        }
    }
    return true;
}

bool AllClose(const Array& a, const Array& b, const Scalar& atol, const Scalar& rtol) {
    if (a.total_size() != b.total_size()) {
        throw DimensionError("cannot compare Arrays of different sizes");
    }
    if (a.dtype() != b.dtype()) {
        throw DtypeError("cannot compare Arrays of different Dtypes");
    }
    if (atol.dtype() != a.dtype() || rtol.dtype() != a.dtype()) {
        throw DtypeError("tolerances need to be of the same Dtype as the arrays");
    }
    switch (a.dtype()) {
        case Dtype::kBool:
            return AllCloseImpl<bool>(a, b, atol, rtol);
        case Dtype::kInt8:
            return AllCloseImpl<int8_t>(a, b, atol, rtol);
        case Dtype::kInt16:
            return AllCloseImpl<int16_t>(a, b, atol, rtol);
        case Dtype::kInt32:
            return AllCloseImpl<int32_t>(a, b, atol, rtol);
        case Dtype::kInt64:
            return AllCloseImpl<int64_t>(a, b, atol, rtol);
        case Dtype::kUInt8:
            return AllCloseImpl<uint8_t>(a, b, atol, rtol);
        case Dtype::kFloat32:
            return AllCloseImpl<float>(a, b, atol, rtol);
        case Dtype::kFloat64:
            return AllCloseImpl<double>(a, b, atol, rtol);
        default:
            assert(false);
    }
    return false;
}

}  // namespace testing
}  // namespace xchainer
