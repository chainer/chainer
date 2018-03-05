#include "xchainer/numeric.h"

#include <cmath>

#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/scalar.h"

namespace xchainer {

bool AllClose(const Array& a, const Array& b, double rtol, double atol) {
    if (a.shape() != b.shape()) {
        throw DimensionError("cannot compare Arrays of different shapes");
    }
    if (a.dtype() != b.dtype()) {
        throw DtypeError("cannot compare Arrays of different Dtypes");
    }
    return VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        int64_t total_size = a.GetTotalSize();
        auto* adata = static_cast<const T*>(a.data().get());
        auto* bdata = static_cast<const T*>(b.data().get());

        for (int64_t i = 0; i < total_size; i++) {
            if (std::abs(adata[i] - bdata[i]) > atol + rtol * std::abs(bdata[i])) {
                return false;
            }
        }
        return true;
    });
}

}  // namespace xchainer
