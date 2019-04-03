#include "chainerx/native/op_regist.h"
#include "chainerx/routines/normalization.h"

namespace chainerx {
namespace native {

CHAINERX_REGISTER_OP_NATIVE(BatchNormForwardOp, GenericBatchNormForwardOp);
CHAINERX_REGISTER_OP_NATIVE(BatchNormBackwardOp, GenericBatchNormBackwardOp);
// TODO(hvy): Rename to FixedBatchNormForwardOp.
CHAINERX_REGISTER_OP_NATIVE(FixedBatchNormOp, GenericFixedBatchNormOp);

}  // namespace native
}  // namespace chainerx
