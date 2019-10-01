#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/trigonometric.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/numeric.h"
#include "chainerx/scalar.h"

namespace chainerx {

namespace internal {
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Sin)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Cos)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Tan)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Arcsin)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Arccos)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Arctan)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Arctan2)
}  // namespace internal

namespace native {
namespace {

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(SinKernel, { out = chainerx::Sin(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(CosKernel, { out = chainerx::Cos(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(TanKernel, { out = chainerx::Tan(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ArcsinKernel, { out = chainerx::Arcsin(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ArccosKernel, { out = chainerx::Arccos(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ArctanKernel, { out = chainerx::Arctan(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_BINARY_KERNEL(Arctan2Kernel, { out = chainerx::Arctan2(x1, x2); });

}  // namespace
}  // namespace native
}  // namespace chainerx
