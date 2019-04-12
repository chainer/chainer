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
#include "chainerx/routines/linalg.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T, typename X>
struct TraceImpl {
    using CudaTypeT = cuda_internal::DataType<T>;
    using CudaTypeX = cuda_internal::DataType<X>;

    CHAINERX_HOST_DEVICE void operator()(const int64_t* index, CudaTypeT& out) {
      if (all_zeros) {
        out = CudaTypeT{0};
        return;
      }

      int64_t x_index[kMaxNdim];
      int out_dim_index = 0;
      for (int j = 0; j < x_ndim; j++) {
        if (j != axis1 && j != axis2) {
          x_index[j] = index[out_dim_index];
          out_dim_index++;
        }
      }

      CudaTypeT summation = CudaTypeT{0};
      for (int64_t i = 0; i < num_elements; i++) {
        x_index[axis1] = start_axis1 + i;
        x_index[axis2] = start_axis2 + i;
        summation = summation + static_cast<CudaTypeT>(
           cuda_internal::StorageToDataType<const X>(x_indexable[x_index]));
      }

      out = summation;
    }

    IndexableArray<const X> x_indexable;

    int64_t axis1;
    int64_t axis2;

    int64_t start_axis1;
    int64_t start_axis2;

    int64_t x_ndim;
    int64_t num_elements;

    bool all_zeros;
};

class CudaTraceOp : public TraceOp {
public:
    void Call(
        const Array& x,
        const int64_t offset,
        const int64_t axis1,
        const int64_t axis2,
        Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaTypeT = cuda_internal::DataType<T>;
            VisitDtype(x.dtype(), [&](auto xt) {
                using X = typename decltype(xt)::type;
                int64_t start_axis1 = 0;
                int64_t start_axis2 = 0;
                int64_t x_ndim = x.ndim();
                if (offset > 0) {
                  start_axis2 = offset;
                } else if (offset < 0) {
                  start_axis1 = -offset;
                }

                const Shape& x_shape = x.shape();
                bool all_zeros = start_axis1 >= x_shape[axis1] || start_axis2 >= x_shape[axis2];
                int64_t num_elements = std::min(
                    x_shape[axis1] - start_axis1, x_shape[axis2] - start_axis2);
                
                ElementwiseWithIndex<T>(TraceImpl<T, X>{
                  IndexableArray<const X>{x}, 
                  axis1, 
                  axis2, 
                  start_axis1, 
                  start_axis2, 
                  x_ndim, 
                  num_elements,
                  all_zeros,
                }, out);
            });
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(TraceOp, CudaTraceOp);

} // namespace
} // namespace cuda
} // namespace chainerx
