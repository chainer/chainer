#include "xchainer/native/native_device.h"

#include <cassert>
#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/native/elementwise.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace native {

namespace {

void DotCheckNdim(int8_t ndim) {
    // TODO(sonots): Support ndim >= 2
    if (ndim != 2) {
        throw DimensionError{"XChainer dot supports only 2-dimensional arrays."};
    }
}

}  // namespace
void NativeDevice::Dot(const Array& a, const Array& b, const Array& out) {
    CheckDevicesCompatible(a, b, out);
    out.Fill(0);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> a_iarray{a};
        IndexableArray<const T> b_iarray{b};
        IndexableArray<T> out_iarray{out};

        // We have to check iarray ndim, otherwise clang-tidy fails bound-checking.
        DotCheckNdim(a_iarray.ndim());
        DotCheckNdim(b_iarray.ndim());
        DotCheckNdim(out_iarray.ndim());

        int64_t m = a.shape()[0];
        int64_t k = a.shape()[1];
        int64_t n = b.shape()[1];
        assert(b.shape()[0] == k);
        assert(out.shape()[0] == m);
        assert(out.shape()[1] == n);

        // TODO(beam2d): Use BLAS.
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t l = 0; l < k; ++l) {
                int64_t a_i_l[] = {i, l};
                T a_value = a_iarray[a_i_l];
                for (int64_t j = 0; j < n; ++j) {
                    int64_t out_i_j[] = {i, j};
                    int64_t b_l_j[] = {l, j};
                    out_iarray[out_i_j] += a_value * b_iarray[b_l_j];
                }
            }
        }
    });
}

}  // namespace native
}  // namespace xchainer
