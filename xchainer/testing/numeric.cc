#include "xchainer/testing/numeric.h"

#include <cassert>

#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace testing {

bool AllClose(const Array& a, const Array& b, double rtol, double atol) {
    if (a.shape() != b.shape()) {
        throw DimensionError("cannot compare Arrays of different shapes");
    }
    if (a.dtype() != b.dtype()) {
        throw DtypeError("cannot compare Arrays of different Dtypes");
    }
    return VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        auto total_size = a.shape().total_size();
        auto adata = static_cast<const T*>(a.data().get());
        auto bdata = static_cast<const T*>(b.data().get());
        auto at = static_cast<const T>(atol);
        auto rt = static_cast<const T>(rtol);
        for (decltype(total_size) i = 0; i < total_size; i++) {
            if (std::abs(adata[i] - bdata[i]) > at + rt * std::abs(bdata[i])) {
                return false;
            }
        }
        return true;
    });
}

}  // namespace testing
}  // namespace xchainer
