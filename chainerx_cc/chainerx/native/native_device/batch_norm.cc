#include "chainerx/native/op_regist.h"
#include "chainerx/routines/normalization.h"

namespace chainerx {
namespace native {

CHAINERX_REGISTER_OP_NATIVE(BatchNormForwardOp, GenericBatchNormForwardOp);
CHAINERX_REGISTER_OP_NATIVE(BatchNormBackwardOp, GenericBatchNormBackwardOp);
CHAINERX_REGISTER_OP_NATIVE(FixedBatchNormForwardOp, GenericFixedBatchNormForwardOp);

}  // namespace native
}  // namespace chainerx
