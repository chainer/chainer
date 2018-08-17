#include "xchainer/numeric.h"

#include <cmath>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"

namespace xchainer {

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
            T ai = a_iarray[it];
            T bi = b_iarray[it];
            if (equal_nan && std::isnan(ai) && std::isnan(bi)) {
                // nop
            } else if (
                    std::isnan(ai) || std::isnan(bi) ||
                    std::abs(static_cast<double>(ai) - static_cast<double>(bi)) > atol + rtol * std::abs(static_cast<double>(bi))) {
                return false;
            }
        }
        return true;
    });
}

}  // namespace xchainer
