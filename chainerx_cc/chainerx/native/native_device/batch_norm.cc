#include "chainerx/kernels/normalization.h"
#include "chainerx/native/op_regist.h"

namespace chainerx {
namespace native {

CHAINERX_NATIVE_REGISTER_OP(BatchNormOp, GenericBatchNormOp);
CHAINERX_NATIVE_REGISTER_OP(BatchNormGradOp, GenericBatchNormGradOp);
CHAINERX_NATIVE_REGISTER_OP(FixedBatchNormOp, GenericFixedBatchNormOp);

}  // namespace native
}  // namespace chainerx
