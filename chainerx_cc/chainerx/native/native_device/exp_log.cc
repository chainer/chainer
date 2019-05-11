#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/math.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/numeric.h"

namespace chainerx {
namespace native {
namespace {

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ExpKernel, { out = chainerx::Exp(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(LogKernel, { out = chainerx::Log(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(Log10Kernel, { out = chainerx::Log10(x); });

}  // namespace
}  // namespace native
}  // namespace chainerx
