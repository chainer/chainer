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
        throw DimensionError("cannot compare Arrays of different shapes");
    }
    if (a.dtype() != b.dtype()) {
        throw DtypeError("cannot compare Arrays of different Dtypes");
    }

    Array a_native = a.ToNative();
    Array b_native = b.ToNative();

    return VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> a_iarray{a_native};
        IndexableArray<const T> b_iarray{b_native};
        Indexer indexer{a_native.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            const T& ai = a_iarray[indexer];
            const T& bi = b_iarray[indexer];
            if (std::isnan(ai) || std::isnan(bi)) {
                // If either is NaN, consider them close only if equal_nan is true and they are both NaN.
                if (!equal_nan || (std::isnan(ai) != std::isnan(bi))) {
                    return false;
                }
            } else if (std::abs(ai - bi) > atol + rtol * std::abs(bi)) {
                return false;
            }
        }
        return true;
    });
}

}  // namespace xchainer
