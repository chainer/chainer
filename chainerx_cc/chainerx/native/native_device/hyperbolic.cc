#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/hyperbolic.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/numeric.h"
#include "chainerx/scalar.h"

namespace chainerx {

namespace internal {
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Sinh)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Cosh)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Tanh)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Arcsinh)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Arccosh)
}  // namespace internal

namespace native {
namespace {

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(SinhKernel, { out = chainerx::Sinh(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(CoshKernel, { out = chainerx::Cosh(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(TanhKernel, { out = chainerx::Tanh(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ArcsinhKernel, { out = chainerx::Arcsinh(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ArccoshKernel, { out = chainerx::Arccosh(x); });

}  // namespace
}  // namespace native
}  // namespace chainerx
