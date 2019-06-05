#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/cuda/numeric.cuh"
#include "chainerx/kernels/rounding.h"

namespace chainerx {
namespace cuda {
namespace {

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(CeilKernel, { out = cuda::Ceil(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(FloorKernel, { out = cuda::Floor(x); });

}  // namespace
}  // namespace cuda
}  // namespace chainerx
