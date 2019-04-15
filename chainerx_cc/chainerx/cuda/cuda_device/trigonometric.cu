#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/numeric.cuh"
#include "chainerx/cuda/op_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/numeric.h"
#include "chainerx/routines/math.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {
namespace {

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(Sin, { out = cuda::Sin(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(Cos, { out = cuda::Cos(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(Tan, { out = cuda::Tan(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(Arcsin, { out = cuda::Arcsin(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(Arccos, { out = cuda::Arccos(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(Arctan, { out = cuda::Arctan(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(Sinh, { out = cuda::Sinh(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(Cosh, { out = cuda::Cosh(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(Arcsinh, { out = cuda::Arcsinh(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(Arccosh, { out = cuda::Arccosh(x); });

}  // namespace
}  // namespace cuda
}  // namespace chainerx
