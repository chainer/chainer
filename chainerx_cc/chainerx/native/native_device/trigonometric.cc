#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/native_device/std_ops.h"
#include "chainerx/native/op_regist.h"
#include "chainerx/numeric.h"
#include "chainerx/routines/math.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace native {
namespace {

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(Sin, { out = chainerx::Sin(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(Cos, { out = chainerx::Cos(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(Tan, { out = chainerx::Tan(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(Arcsin, { out = chainerx::Arcsin(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(Arccos, { out = chainerx::Arccos(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(Arctan, { out = chainerx::Arctan(x); });

}  // namespace
}  // namespace native
}  // namespace chainerx
