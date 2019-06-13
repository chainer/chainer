#include "chainerx/native/op_regist.h"
#include "chainerx/routines/normalization.h"

namespace chainerx {
namespace native {

CHAINERX_REGISTER_OP_NATIVE(BatchNormOp, GenericBatchNormOp);
CHAINERX_REGISTER_OP_NATIVE(BatchNormGradOp, GenericBatchNormGradOp);
CHAINERX_REGISTER_OP_NATIVE(FixedBatchNormOp, GenericFixedBatchNormOp);

}  // namespace native
}  // namespace chainerx
