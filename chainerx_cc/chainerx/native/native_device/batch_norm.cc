#include "chainerx/kernels/normalization.h"
#include "chainerx/native/kernel_regist.h"

namespace chainerx {
namespace native {

CHAINERX_NATIVE_REGISTER_KERNEL(BatchNormKernel, GenericBatchNormKernel);
CHAINERX_NATIVE_REGISTER_KERNEL(BatchNormGradKernel, GenericBatchNormGradKernel);
CHAINERX_NATIVE_REGISTER_KERNEL(FixedBatchNormKernel, GenericFixedBatchNormKernel);

}  // namespace native
}  // namespace chainerx
