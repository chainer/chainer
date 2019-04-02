#include "chainerx/native/op_regist.h"
#include "chainerx/routines/normalization.h"

namespace chainerx {
namespace native {

class NativeBatchNormOp : public GenericBatchNormOp {};

CHAINERX_REGISTER_OP_NATIVE(BatchNormOp, NativeBatchNormOp);

class NativeFixedBatchNormOp : public GenericFixedBatchNormOp {};

CHAINERX_REGISTER_OP_NATIVE(FixedBatchNormOp, NativeFixedBatchNormOp);

}  // namespace native
}  // namespace chainerx
