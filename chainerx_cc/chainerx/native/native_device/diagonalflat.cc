#include "chainerx/native/native_device.h"

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/macro.h"
#include "chainerx/native/data_type.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace native {
namespace {

class NativeDiagonalflatKernel : public DiagonalflatKernel {
public:
    void Call(const Array& x, int64_t offset, int64_t axis1, int64_t axis2, Array& out) {
        x.device().CheckDevicesCompatible(x, out);
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                explicit Impl(const Array& a) : x_indexable{a} {}
                void operator()(const int64_t* index, T& out) {
                    int64_t element_index = (offset >= 0) ? index[axis1] : index[axis2];
                    bool valid_element = (offset >= 0 && index[axis2] == (index[axis1] + offset)) ||
                                         (offset < 0 && index[axis1] == (index[axis2] - offset));

                    if (valid_element) {
                        int64_t j = 0;
                        for (int64_t i = 0; i < out_ndim; i++) {
                            if (i != axis1 && i != axis2) {
                                x_index[j++] = index[i];
                            }
                        }
                        x_index[j] = element_index;

                        out = static_cast<T>(native::StorageToDataType<const T>(x_indexable[x_index.data()]));
                    } else {
                        out = T{0};
                    }
                }

                int64_t axis1;
                int64_t axis2;
                int64_t offset;

                IndexableArray<const T> x_indexable;
                std::vector<int64_t> x_index;

                int64_t out_ndim;
            };

            Impl impl{x};
            impl.axis1 = axis1;
            impl.axis2 = axis2;
            impl.offset = offset;

            impl.x_index.resize(x.ndim());

            impl.out_ndim = x.ndim() + 1;

            ElementwiseWithIndex<T>(std::move(impl), out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(DiagonalflatKernel, NativeDiagonalflatKernel);

}  // namespace
}  // namespace native
}  // namespace chainerx
