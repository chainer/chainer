#include "chainerx/cuda/cuda_device.h"

#include <cstdint>
#include <mutex>
#include <type_traits>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.hpp>

#include "chainerx/array.h"
#include "chainerx/backend_util.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/float16.cuh"
#include "chainerx/cuda/op_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/float16.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/indexing.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct DiagonalflatImpl {
    using CudaTypeT = cuda_internal::DataType<T>;

    explicit DiagonalflatImpl(const Array& a) : x_indexable{a} {}
    __device__ void operator()(const int64_t* index, CudaTypeT& out) {
        int64_t x_index[kMaxNdim];
        int64_t element_index = (offset >= 0) ? index[axis1] : index[axis2];
        bool valid_element =
                (offset >= 0 && index[axis2] == (index[axis1] + offset)) || (offset < 0 && index[axis1] == (index[axis2] - offset));

        if (valid_element) {
            int64_t j = 0;
            for (int64_t i = 0; i < out_ndim; i++) {
                if (i != axis1 && i != axis2) {
                    x_index[j++] = index[i];
                }
            }
            x_index[j] = element_index;
            out = static_cast<CudaTypeT>(cuda_internal::StorageToDataType<const T>(x_indexable[x_index]));
        } else {
            out = CudaTypeT{0};
        }
    }

    int64_t axis1;
    int64_t axis2;
    int64_t offset;

    int64_t x_ndim;
    IndexableArray<const T> x_indexable;

    int64_t out_ndim;
};

class CudaDiagonalflatOp : public DiagonalflatOp {
public:
    void Call(const Array& x, int64_t offset, int64_t axis1, int64_t axis2, int64_t, Array& out) {
        x.device().CheckDevicesCompatible(x, out);
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;

            DiagonalflatImpl<T> impl{x};
            impl.axis1 = axis1;
            impl.axis2 = axis2;
            impl.offset = offset;

            impl.x_ndim = x.ndim();
            impl.out_ndim = impl.x_ndim + 1;

            ElementwiseWithIndex<T>(std::move(impl), out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(DiagonalflatOp, CudaDiagonalflatOp);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
