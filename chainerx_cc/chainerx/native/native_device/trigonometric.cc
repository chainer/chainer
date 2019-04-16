#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/op_regist.h"
#include "chainerx/numeric.h"
#include "chainerx/routines/math.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace native {
namespace {

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(SinOp, { out = chainerx::Sin(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(CosOp, { out = chainerx::Cos(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(TanOp, { out = chainerx::Tan(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(ArcsinOp, { out = chainerx::Arcsin(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(ArccosOp, { out = chainerx::Arccos(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(ArctanOp, { out = chainerx::Arctan(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(SinhOp, { out = chainerx::Sinh(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(CoshOp, { out = chainerx::Cosh(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(ArcsinhOp, { out = chainerx::Arcsinh(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(ArccoshOp, { out = chainerx::Arccosh(x); });

}  // namespace
}  // namespace native
}  // namespace chainerx
