#include "chainerx/kernels/normalization.h"
#include "chainerx/native/kernel_regist.h"

namespace chainerx {

namespace internal {
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(BatchNorm)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(BatchNormGrad)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(FixedBatchNorm)
}  // namespace internal

namespace native {

CHAINERX_NATIVE_REGISTER_KERNEL(BatchNormKernel, GenericBatchNormKernel);
CHAINERX_NATIVE_REGISTER_KERNEL(BatchNormGradKernel, GenericBatchNormGradKernel);
CHAINERX_NATIVE_REGISTER_KERNEL(FixedBatchNormKernel, GenericFixedBatchNormKernel);

}  // namespace native
}  // namespace chainerx
