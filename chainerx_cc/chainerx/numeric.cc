#include "chainerx/numeric.h"

#include <cmath>

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/native/data_type.h"

namespace chainerx {

bool AllClose(const Array& a, const Array& b, double rtol, double atol, bool equal_nan) {
    if (a.shape() != b.shape()) {
        throw DimensionError{"Cannot compare Arrays of different shapes: ", a.shape(), ", ", b.shape()};
    }
    if (a.dtype() != b.dtype()) {
        throw DtypeError{"Cannot compare Arrays of different Dtypes: ", a.dtype(), ", ", b.dtype()};
    }

    Array a_native = a.ToNative();
    Array b_native = b.ToNative();

    return VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> a_iarray{a_native};
        IndexableArray<const T> b_iarray{b_native};
        Indexer<> indexer{a_native.shape()};

        for (auto it = indexer.It(0); it; ++it) {
            T ai = native::StorageToDataType<const T>(a_iarray[it]);
            T bi = native::StorageToDataType<const T>(b_iarray[it]);
            if (equal_nan && IsNan(ai) && IsNan(bi)) {
                // nop
            } else if (
                    IsNan(ai) || IsNan(bi) ||
                    std::abs(static_cast<double>(ai) - static_cast<double>(bi)) > atol + rtol * std::abs(static_cast<double>(bi))) {
                return false;
            }
        }
        return true;
    });
}

}  // namespace chainerx
