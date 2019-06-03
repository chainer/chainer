#include "chainerx/cuda/cuda_device.h"

#include <cstdint>
#include <mutex>
#include <type_traits>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cusolver.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/float16.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/float16.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/kernels/math.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/math.h"

namespace chainerx {
namespace cuda {

class CudaQRKernel : public QRKernel {
public:
    std::tuple<Array, Array> Call(const Array& a) override {
        Device& device = a.device();
        Dtype dtype = internal::GetMathResultDtype(a.dtype());
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);

        throw NotImplementedError("QR decomposition is not yet implemented for cuda device");

    }
};

CHAINERX_CUDA_REGISTER_KERNEL(QRKernel, CudaQRKernel);

}  // namespace cuda
}  // namespace chainerx
