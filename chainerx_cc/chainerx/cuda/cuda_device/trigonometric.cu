#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/cuda/numeric.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/trigonometric.h"
#include "chainerx/numeric.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {
namespace {

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(SinKernel, { out = cuda::Sin(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(CosKernel, { out = cuda::Cos(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(TanKernel, { out = cuda::Tan(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ArcsinKernel, { out = cuda::Arcsin(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ArccosKernel, { out = cuda::Arccos(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ArctanKernel, { out = cuda::Arctan(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_BINARY_KERNEL(Arctan2Kernel, { out = cuda::Arctan2(x1, x2); });

}  // namespace
}  // namespace cuda
}  // namespace chainerx
