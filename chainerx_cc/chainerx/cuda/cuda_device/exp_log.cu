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
#include "chainerx/kernels/explog.h"

namespace chainerx {
namespace cuda {
namespace {

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ExpKernel, { out = cuda::Exp(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(Expm1Kernel, { out = cuda::Expm1(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(Exp2Kernel, { out = cuda::Exp2(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(LogKernel, { out = cuda::Log(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(Log10Kernel, { out = cuda::Log10(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(Log1pKernel, { out = cuda::Log1p(x); });

}  // namespace
}  // namespace cuda
}  // namespace chainerx
