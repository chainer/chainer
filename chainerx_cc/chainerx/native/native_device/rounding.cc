#include "chainerx/kernels/rounding.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/numeric.h"

namespace chainerx {

namespace internal {
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Ceil)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Floor)
}  // namespace internal

namespace native {
namespace {

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(CeilKernel, { out = chainerx::Ceil(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(FloorKernel, { out = chainerx::Floor(x); });

}  // namespace
}  // namespace native
}  // namespace chainerx
