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
#include "chainerx/routines/linalg.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace native {
namespace {

class NativeTraceOp : public TraceOp {
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
                        if (all_zeros) {
                            out = T{0};
                            return;
                        }

                        int out_dim_index = 0;
                        for (int j = 0; j < x_ndim; j++) {
                            if (j != axis1 && j != axis2) {
                                x_index[j] = index[out_dim_index];
                                out_dim_index++;
                            }
                        }

                        T summation = T{0};
                        for (int64_t i = 0; i < num_elements; i++) {
                            x_index[axis1] = start_axis1 + i;
                            x_index[axis2] = start_axis2 + i;

                            summation = summation + static_cast<T>(native::StorageToDataType<const X>(x_indexable[x_index.data()]));
                        }

                        out = summation;
                    }

                    int64_t axis1;
                    int64_t axis2;

                    int64_t start_axis1;
                    int64_t start_axis2;

                    int64_t x_ndim;
                    IndexableArray<const X> x_indexable;
                    std::vector<int64_t> x_index;

                    bool all_zeros;
                    int64_t num_elements;
                };

                Impl impl{x};
                impl.axis1 = axis1;
                impl.axis2 = axis2;
                impl.start_axis1 = 0;
                impl.start_axis2 = 0;
                impl.x_ndim = x.ndim();
                impl.x_index.resize(x.ndim());
                impl.all_zeros = false;
                if (offset > 0) {
                    impl.start_axis2 = offset;
                } else if (offset < 0) {
                    impl.start_axis1 = -offset;
                }

                const Shape& x_shape = x.shape();
                impl.all_zeros = impl.start_axis1 >= x_shape[axis1] || impl.start_axis2 >= x_shape[axis2];
                if (!impl.all_zeros) {
                    impl.num_elements = std::min(x_shape[axis1] - impl.start_axis1, x_shape[axis2] - impl.start_axis2);
                } else {
                    impl.num_elements = 0;
                }

                ElementwiseWithIndex<T>(impl, out);
            });
        });
    }
};

CHAINERX_REGISTER_OP_NATIVE(TraceOp, NativeTraceOp);

}  // namespace
}  // namespace native
}  // namespace chainerx

