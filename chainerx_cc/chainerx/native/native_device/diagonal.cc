#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

#ifdef CHAINERX_ENABLE_BLAS
#include <cblas.h>
#endif  // CHAINERX_ENABLE_BLAS

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/macro.h"
#include "chainerx/native/data_type.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/op_regist.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace native {
namespace {

class NativeDiagonalOp : public DiagonalOp {
public:
    void Call(const Array& x, const int64_t offset, const int64_t axis1, const int64_t axis2, Array& out) {
        x.device().CheckDevicesCompatible(x, out);
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            VisitDtype(x.dtype(), [&](auto xt) {
                using X = typename decltype(xt)::type;
                struct Impl {
                    explicit Impl(const Array& a) : x_indexable{a} {}
                    void operator()(const int64_t* index, T& out) {
                        int out_dim_index = 0;
                        for (int j = 0; j < x_indexable.ndim(); j++) {
                            if (j != axis1 && j != axis2) {
                                x_index[j] = index[out_dim_index];
                                out_dim_index++;
                            }
                        }

                        x_index[axis1] = start_axis1 + index[out_dim_index];
                        x_index[axis2] = start_axis2 + index[out_dim_index];

                        out = static_cast<T>(native::StorageToDataType<const X>(x_indexable[x_index.data()]));
                    }

                    int64_t axis1;
                    int64_t axis2;

                    int64_t start_axis1;
                    int64_t start_axis2;

                    IndexableArray<const X> x_indexable;
                    std::vector<int64_t> x_index;

                    int64_t num_elements;
                };

                Impl impl{x};
                impl.axis1 = axis1;
                impl.axis2 = axis2;
                impl.start_axis1 = (offset < 0) ? -offset : 0;
                impl.start_axis2 = (offset > 0) ? offset : 0;
                impl.x_index.resize(x.ndim());

                const Shape& x_shape = x.shape();
                impl.num_elements = std::max(0l,
                    std::min(x_shape[axis1] - impl.start_axis1, x_shape[axis2] - impl.start_axis2));

                ElementwiseWithIndex<T>(std::move(impl), out);
            });
        });
    }
};

CHAINERX_REGISTER_OP_NATIVE(DiagonalOp, NativeDiagonalOp);

}  // namespace
}  // namespace native
}  // namespace chainerx
