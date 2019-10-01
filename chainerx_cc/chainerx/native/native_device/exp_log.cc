#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/explog.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/numeric.h"

namespace chainerx {

namespace internal {
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Erf)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Exp)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Expm1)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Exp2)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Log)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Log10)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Log2)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Log1p)
}  // namespace internal

namespace native {
namespace {

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ErfKernel, { out = chainerx::Erf(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ExpKernel, { out = chainerx::Exp(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(Expm1Kernel, { out = chainerx::Expm1(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(Exp2Kernel, { out = chainerx::Exp2(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(LogKernel, { out = chainerx::Log(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(Log10Kernel, { out = chainerx::Log10(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(Log2Kernel, { out = chainerx::Log2(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(Log1pKernel, { out = chainerx::Log1p(x); });

}  // namespace
}  // namespace native
}  // namespace chainerx
