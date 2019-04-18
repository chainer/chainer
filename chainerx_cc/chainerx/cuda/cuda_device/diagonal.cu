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
struct DiagonalImpl {
    using CudaTypeT = cuda_internal::DataType<T>;

    explicit DiagonalImpl(const Array& a) : x_indexable{a} {}
    __device__ void operator()(const int64_t* index, CudaTypeT& out) {
        int64_t x_index[kMaxNdim];
        int out_dim_index = 0;
        for (int j = 0; j < x_ndim; j++) {
            if (j != axis1 && j != axis2) {
                x_index[j] = index[out_dim_index];
                out_dim_index++;
            }
        }

        x_index[axis1] = start_axis1 + index[out_dim_index];
        x_index[axis2] = start_axis2 + index[out_dim_index];

        out = static_cast<CudaTypeT>(cuda_internal::StorageToDataType<const T>(x_indexable[x_index]));
    }

    int64_t axis1;
    int64_t axis2;

    int64_t start_axis1;
    int64_t start_axis2;

    int64_t x_ndim;
    IndexableArray<const T> x_indexable;

    int64_t num_elements;
};

class CudaDiagonalOp : public DiagonalOp {
public:
    void Call(const Array& x, const int64_t offset, const int64_t axis1, const int64_t axis2, Array& out) {
        x.device().CheckDevicesCompatible(x, out);
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;

            DiagonalImpl<T> impl{x};
            impl.axis1 = axis1;
            impl.axis2 = axis2;
            impl.start_axis1 = (offset < 0) ? -offset : 0;
            impl.start_axis2 = (offset > 0) ? offset : 0;
            impl.x_ndim = x.ndim();

            const Shape& x_shape = x.shape();
            impl.num_elements = std::max(0l,
                std::min(x_shape[axis1] - impl.start_axis1, x_shape[axis2] - impl.start_axis2));

            std::vector<int64_t> out_shape;
            for (int i = 0; i < x.ndim(); i++) {
                if (i != axis1 && i != axis2) out_shape.push_back(x_shape[i]);
            }
            out_shape.push_back(impl.num_elements);

            out = Empty(Shape{out_shape}, x.dtype(), x.device());

            ElementwiseWithIndex<T>(std::move(impl), out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(DiagonalOp, CudaDiagonalOp);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
