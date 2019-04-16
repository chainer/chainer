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
#include "chainerx/kernels/math.h"
#include "chainerx/numeric.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {
namespace {

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(SinOp, { out = cuda::Sin(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(CosOp, { out = cuda::Cos(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(TanOp, { out = cuda::Tan(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(ArcsinOp, { out = cuda::Arcsin(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(ArccosOp, { out = cuda::Arccos(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(ArctanOp, { out = cuda::Arctan(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(SinhOp, { out = cuda::Sinh(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(CoshOp, { out = cuda::Cosh(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(ArcsinhOp, { out = cuda::Arcsinh(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(ArccoshOp, { out = cuda::Arccosh(x); });

}  // namespace
}  // namespace cuda
}  // namespace chainerx
