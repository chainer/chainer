// TODO(niboshi): data_type.cuh and elementwise.cuh should be included in kernel_regist.h (after renamed to .cuh)
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/elementwise.cuh"
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
