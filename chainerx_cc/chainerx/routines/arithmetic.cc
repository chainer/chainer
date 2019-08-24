#include "chainerx/routines/arithmetic.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/enum.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/arithmetic.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/explog.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/statistics.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {

Array Negative(const Array& x) {
    if (x.dtype() == Dtype::kBool) {
        throw DtypeError{"Cannot negate a boolean array."};
    }
    return Multiply(x, Scalar{-1, GetKind(x.dtype())});
}

namespace {

void CheckArithmeticDtypes(DtypeKind kind1, DtypeKind kind2, bool is_multiply) {
    if (!is_multiply && (kind1 == DtypeKind::kBool || kind2 == DtypeKind::kBool)) {
        throw DtypeError{"Boolean arguments are not allowed in arithmetic functions."};
    }
}

Dtype GetArithmeticResultDtype(const Array& x1, const Array& x2, bool is_multiply = false) {
    CheckArithmeticDtypes(GetKind(x1.dtype()), GetKind(x2.dtype()), is_multiply);
    return ResultType(x1, x2);
}

Dtype GetArithmeticResultDtype(const Array& x1, Scalar x2, bool is_multiply = false) {
    CheckArithmeticDtypes(GetKind(x1.dtype()), x2.kind(), is_multiply);
    return ResultType(x1, x2);
}

Dtype GetArithmeticResultDtype(Scalar x1, const Array& x2, bool is_multiply = false) {
    CheckArithmeticDtypes(x1.kind(), GetKind(x2.dtype()), is_multiply);
    return ResultType(x1, x2);
}

void CheckInplaceArithmeticDtypes(DtypeKind out_kind, DtypeKind in_kind, bool is_multiply = false) {
    CheckArithmeticDtypes(out_kind, in_kind, is_multiply);
    if (out_kind != DtypeKind::kFloat && in_kind == DtypeKind::kFloat) {
        throw DtypeError{"In-place assignment of float values into non-float array is not allowed."};
    }
    if (is_multiply && out_kind == DtypeKind::kBool && in_kind != DtypeKind::kBool) {
        throw DtypeError{"In-place assignment of numerical values into bool array is not allowed."};
    }
}

void CheckInplaceArithmeticDtypes(const Array& x1, const Array& x2, bool is_multiply = false) {
    CheckInplaceArithmeticDtypes(GetKind(x1.dtype()), GetKind(x2.dtype()), is_multiply);
}

void CheckInplaceArithmeticDtypes(const Array& x1, Scalar x2, bool is_multiply = false) {
    CheckInplaceArithmeticDtypes(GetKind(x1.dtype()), x2.kind(), is_multiply);
}

}  // namespace

void AddImpl(const Array& x1, const Array& x2, const Array& out) {
    CheckEqual(x1.shape(), x2.shape());

    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<AddKernel>(x1, x2, out);
    }

    {
        BackwardBuilder bb{"add", {x1, x2}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([dtype = x1.dtype()](BackwardContext& bctx) {
                const Array& gx = *bctx.output_grad();
                bctx.input_grad() = dtype == gx.dtype() ? gx : gx.AsType(dtype);
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([dtype = x2.dtype()](BackwardContext& bctx) {
                const Array& gx = *bctx.output_grad();
                bctx.input_grad() = dtype == gx.dtype() ? gx : gx.AsType(dtype);
            });
        }
        bb.Finalize();
    }
}

void AddASImpl(const Array& x1, Scalar x2, const Array& out) {
    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<AddASKernel>(x1, x2, out);
    }

    BackwardBuilder bb{"add_scalar", x1, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
    }
    bb.Finalize();
}

namespace internal {

void IAdd(const Array& x1, const Array& x2) {
    CheckInplaceArithmeticDtypes(x1, x2);
    internal::BroadcastBinaryInplace(&AddImpl, x1, x2);
}

void IAdd(const Array& x1, Scalar x2) {
    CheckInplaceArithmeticDtypes(x1, x2);
    internal::BinaryInplace(&AddASImpl, x1, x2);
}

}  // namespace internal

Array Add(const Array& x1, const Array& x2) { return internal::BroadcastBinary(&AddImpl, x1, x2, GetArithmeticResultDtype(x1, x2)); }

Array Add(const Array& x1, Scalar x2) { return internal::Binary(&AddASImpl, x1, x2, GetArithmeticResultDtype(x1, x2)); }

Array Add(Scalar x1, const Array& x2) { return Add(x2, x1); }

void SubtractImpl(const Array& x1, const Array& x2, const Array& out) {
    CheckEqual(x1.shape(), x2.shape());

    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<SubtractKernel>(x1, x2, out);
    }

    {
        BackwardBuilder bb{"subtract", {x1, x2}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([dtype = x1.dtype()](BackwardContext& bctx) {
                const Array& gx = *bctx.output_grad();
                bctx.input_grad() = dtype == gx.dtype() ? gx : gx.AsType(dtype);
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([dtype = x2.dtype()](BackwardContext& bctx) {
                Array gx = -*bctx.output_grad();
                bctx.input_grad() = dtype == gx.dtype() ? std::move(gx) : gx.AsType(dtype);
            });
        }
        bb.Finalize();
    }
}

void SubtractASImpl(const Array& x1, Scalar x2, const Array& out) {
    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<SubtractASKernel>(x1, x2, out);
    }

    BackwardBuilder bb{"subtract_scalar", x1, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
    }
    bb.Finalize();
}

namespace internal {

void ISubtract(const Array& x1, const Array& x2) {
    CheckInplaceArithmeticDtypes(x1, x2);
    internal::BroadcastBinaryInplace(SubtractImpl, x1, x2);
}

void ISubtract(const Array& x1, Scalar x2) {
    CheckInplaceArithmeticDtypes(x1, x2);
    internal::BinaryInplace(&SubtractASImpl, x1, x2);
}

}  // namespace internal

Array Subtract(const Array& x1, const Array& x2) {
    return internal::BroadcastBinary(&SubtractImpl, x1, x2, GetArithmeticResultDtype(x1, x2));
}

Array Subtract(const Array& x1, Scalar x2) { return internal::Binary(&SubtractASImpl, x1, x2, GetArithmeticResultDtype(x1, x2)); }

Array Subtract(Scalar x1, const Array& x2) {
    // TODO(imanishi): Avoid type casting. This cast is introduced in order to avoid overflow in negative operation.
    // Remove this cast after device implementation of subtract (scalar - array) is added.
    if ((GetKind(x2.dtype()) == DtypeKind::kUInt || GetKind(x2.dtype()) == DtypeKind::kInt) && x1.kind() == DtypeKind::kFloat) {
        Array x2_cast = x2.AsType(ResultType(x1, x2));
        return Add(-x2_cast, x1);
    }
    return Add(-x2, x1);
}

void MultiplyImpl(const Array& x1, const Array& x2, const Array& out) {
    CheckEqual(x1.shape(), x2.shape());

    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<MultiplyKernel>(x1, x2, out);
    }

    {
        BackwardBuilder bb{"multiply", {x1, x2}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([x2_tok = bb.RetainInput(1), dtype = x1.dtype()](BackwardContext& bctx) {
                const Array& x2 = bctx.GetRetainedInput(x2_tok);
                Array gx = *bctx.output_grad() * x2;
                bctx.input_grad() = dtype == gx.dtype() ? std::move(gx) : gx.AsType(dtype);
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([x1_tok = bb.RetainInput(0), dtype = x2.dtype()](BackwardContext& bctx) {
                const Array& x1 = bctx.GetRetainedInput(x1_tok);
                Array gx = *bctx.output_grad() * x1;
                bctx.input_grad() = dtype == gx.dtype() ? std::move(gx) : gx.AsType(dtype);
            });
        }
        bb.Finalize();
    }
}

void MultiplyASImpl(const Array& x1, Scalar x2, const Array& out) {
    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<MultiplyASKernel>(x1, x2, out);
    }

    BackwardBuilder bb{"multiply_scalar", x1, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x2](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad() * x2; });
    }
    bb.Finalize();
}

namespace internal {

void IMultiply(const Array& x1, const Array& x2) {
    CheckInplaceArithmeticDtypes(x1, x2, true);
    internal::BroadcastBinaryInplace(&MultiplyImpl, x1, x2);
}

void IMultiply(const Array& x1, Scalar x2) {
    CheckInplaceArithmeticDtypes(x1, x2, true);
    internal::BinaryInplace(&MultiplyASImpl, x1, x2);
}

}  // namespace internal

Array Multiply(const Array& x1, const Array& x2) {
    return internal::BroadcastBinary(&MultiplyImpl, x1, x2, GetArithmeticResultDtype(x1, x2, true));
}

Array Multiply(const Array& x1, Scalar x2) { return internal::Binary(&MultiplyASImpl, x1, x2, GetArithmeticResultDtype(x1, x2, true)); }

Array Multiply(Scalar x1, const Array& x2) { return Multiply(x2, x1); }

void FloorDivideImpl(const Array& x1, const Array& x2, const Array& out) {
    CheckEqual(x1.shape(), x2.shape());

    NoBackpropModeScope scope{};
    x1.device().backend().CallKernel<FloorDivideKernel>(x1, x2, out);
}

void FloorDivideASImpl(const Array& x1, Scalar x2, const Array& out) {
    NoBackpropModeScope scope{};
    x1.device().backend().CallKernel<FloorDivideASKernel>(x1, x2, out);
}

void FloorDivideSAImpl(Scalar x1, const Array& x2, const Array& out) {
    NoBackpropModeScope scope{};
    x2.device().backend().CallKernel<FloorDivideSAKernel>(x1, x2, out);
}

namespace internal {

void IFloorDivide(const Array& x1, const Array& x2) {
    CheckInplaceArithmeticDtypes(x1, x2);
    internal::BroadcastBinaryInplace(&FloorDivideImpl, x1, x2);
}

void IFloorDivide(const Array& x1, Scalar x2) {
    CheckInplaceArithmeticDtypes(x1, x2);
    internal::BinaryInplace(&FloorDivideASImpl, x1, x2);
}

}  // namespace internal

Array FloorDivide(const Array& x1, const Array& x2) {
    return internal::BroadcastBinary(&FloorDivideImpl, x1, x2, GetArithmeticResultDtype(x1, x2));
}

Array FloorDivide(const Array& x1, Scalar x2) { return internal::Binary(&FloorDivideASImpl, x1, x2, GetArithmeticResultDtype(x1, x2)); }

Array FloorDivide(Scalar x1, const Array& x2) { return internal::Binary(&FloorDivideSAImpl, x1, x2, GetArithmeticResultDtype(x1, x2)); }

void DivideImpl(const Array& x1, const Array& x2, const Array& out) {
    CheckEqual(x1.shape(), x2.shape());

    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<DivideKernel>(x1, x2, out);
    }

    {
        BackwardBuilder bb{"divide", {x1, x2}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([x2_tok = bb.RetainInput(1), dtype = x1.dtype()](BackwardContext& bctx) {
                const Array& x2 = bctx.GetRetainedInput(x2_tok);
                Array gx = *bctx.output_grad() / x2;
                bctx.input_grad() = dtype == gx.dtype() ? std::move(gx) : gx.AsType(dtype);
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([x1_tok = bb.RetainInput(0), x2_tok = bb.RetainInput(1), dtype = x2.dtype()](BackwardContext& bctx) {
                const Array& x1 = bctx.GetRetainedInput(x1_tok);
                const Array& x2 = bctx.GetRetainedInput(x2_tok);
                Array gx = -*bctx.output_grad() * x1 / Square(x2);
                bctx.input_grad() = dtype == gx.dtype() ? std::move(gx) : gx.AsType(dtype);
            });
        }
        bb.Finalize();
    }
}

void DivideASImpl(const Array& x1, Scalar x2, const Array& out) {
    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<DivideASKernel>(x1, x2, out);
    }

    BackwardBuilder bb{"divide_array_scalar", x1, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([other = x2](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad() / other; });
    }
    bb.Finalize();
}

void DivideSAImpl(Scalar x1, const Array& x2, const Array& out) {
    {
        NoBackpropModeScope scope{};
        x2.device().backend().CallKernel<DivideSAKernel>(x1, x2, out);
    }

    BackwardBuilder bb{"divide_scalar_array", x2, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([other = x1, x2_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& x2 = bctx.GetRetainedInput(x2_tok);
            bctx.input_grad() = -*bctx.output_grad() * other / Square(x2);
        });
    }
    bb.Finalize();
}

namespace internal {

void ITrueDivide(const Array& x1, const Array& x2) {
    if (GetKind(x1.dtype()) != DtypeKind::kFloat) {
        throw DtypeError{"Integer inplace-division is not allowed."};
    }
    CheckInplaceArithmeticDtypes(x1, x2);
    internal::BroadcastBinaryInplace(&DivideImpl, x1, x2);
}

void ITrueDivide(const Array& x1, Scalar x2) {
    if (GetKind(x1.dtype()) != DtypeKind::kFloat) {
        throw DtypeError{"Integer inplace-division is not allowed."};
    }
    CheckInplaceArithmeticDtypes(x1, x2);
    internal::BinaryInplace(&DivideASImpl, x1, x2);
}

void IDivide(const Array& x1, const Array& x2) { ITrueDivide(x1, x2); }

void IDivide(const Array& x1, Scalar x2) { ITrueDivide(x1, x2); }

}  // namespace internal

Array TrueDivide(const Array& x1, const Array& x2) {
    Dtype dtype = GetArithmeticResultDtype(x1, x2);
    if (GetKind(dtype) != DtypeKind::kFloat) {
        dtype = internal::GetDefaultDtype(DtypeKind::kFloat);
    }
    return internal::BroadcastBinary(&DivideImpl, x1, x2, dtype);
}

Array TrueDivide(const Array& x1, Scalar x2) {
    Dtype dtype = GetArithmeticResultDtype(x1, x2);
    if (GetKind(dtype) != DtypeKind::kFloat) {
        dtype = internal::GetDefaultDtype(DtypeKind::kFloat);
    }
    return internal::Binary(&DivideASImpl, x1, x2, dtype);
}

Array TrueDivide(Scalar x1, const Array& x2) {
    Dtype dtype = GetArithmeticResultDtype(x1, x2);
    if (GetKind(dtype) != DtypeKind::kFloat) {
        dtype = internal::GetDefaultDtype(DtypeKind::kFloat);
    }
    return internal::Binary(&DivideSAImpl, x1, x2, dtype);
}

Array Divide(const Array& x1, const Array& x2) { return TrueDivide(x1, x2); }

Array Divide(const Array& x1, Scalar x2) { return TrueDivide(x1, x2); }

Array Divide(Scalar x1, const Array& x2) { return TrueDivide(x1, x2); }

Array Reciprocal(const Array& x) { return Scalar{1, GetKind(x.dtype())} / x; }

void PowerImpl(const Array& x1, const Array& x2, const Array& out) {
    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<PowerKernel>(x1, x2, out);
    }

    BackwardBuilder bb{"power", {x1, x2}, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x1_tok = bb.RetainInput(0), x2_tok = bb.RetainInput(1)](BackwardContext& bctx) {
            const Array& x1 = bctx.GetRetainedInput(x1_tok);
            const Array& x2 = bctx.GetRetainedInput(x2_tok);
            const Array& gin = *bctx.output_grad() * x2 * Power(x1, x2 - Scalar{1, GetKind(x2.dtype())});
            bctx.input_grad() = x1.dtype() != gin.dtype() ? gin.AsType(x1.dtype()) : gin;
        });
    }
    if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
        bt.Define([x1_tok = bb.RetainInput(0), out_tok = bb.RetainOutput(0), x2_dtype = x2.dtype()](BackwardContext& bctx) {
            const Array& x1 = bctx.GetRetainedInput(x1_tok);
            const Array& out = bctx.GetRetainedOutput(out_tok);
            const Array& gin = *bctx.output_grad() * out * Log(x1);
            bctx.input_grad() = x2_dtype != gin.dtype() ? gin.AsType(x2_dtype) : gin;
        });
    }
    bb.Finalize();
}

void PowerASImpl(const Array& x1, Scalar x2, const Array& out) {
    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<PowerASKernel>(x1, x2, out);
    }

    BackwardBuilder bb{"power_array_scalar", x1, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x1_tok = bb.RetainInput(0), x2](BackwardContext& bctx) {
            const Array& x1 = bctx.GetRetainedInput(x1_tok);
            const Array& gin = *bctx.output_grad() * x2 * Power(x1, x2 - Scalar{1, x2.kind()});
            bctx.input_grad() = x1.dtype() != gin.dtype() ? gin.AsType(x1.dtype()) : gin;
        });
    }
    bb.Finalize();
}

void PowerSAImpl(Scalar x1, const Array& x2, const Array& out) {
    {
        NoBackpropModeScope scope{};
        x2.device().backend().CallKernel<PowerSAKernel>(x1, x2, out);
    }

    BackwardBuilder bb{"power_scalar_array", x2, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([out_tok = bb.RetainOutput(0), x1, x2_dtype = x2.dtype()](BackwardContext& bctx) {
            const Array& out = bctx.GetRetainedOutput(out_tok);
            const Array& gin = *bctx.output_grad() * out * std::log(static_cast<double>(x1));
            bctx.input_grad() = x2_dtype != gin.dtype() ? gin.AsType(x2_dtype) : gin;
        });
    }
    bb.Finalize();
}

Array Power(const Array& x1, const Array& x2) { return internal::BroadcastBinary(&PowerImpl, x1, x2, GetArithmeticResultDtype(x1, x2)); }

Array Power(const Array& x1, Scalar x2) { return internal::Binary(&PowerASImpl, x1, x2, GetArithmeticResultDtype(x1, x2)); }

Array Power(Scalar x1, const Array& x2) { return internal::Binary(&PowerSAImpl, x1, x2, GetArithmeticResultDtype(x1, x2)); }

void ModImpl(const Array& x1, const Array& x2, const Array& out) {
    CheckEqual(x1.shape(), x2.shape());

    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<ModAAKernel>(x1, x2, out);
    }

    {
        BackwardBuilder bb{"mod", {x1, x2}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([dtype = x1.dtype()](BackwardContext& bctx) {
                const Array& gx = *bctx.output_grad();
                bctx.input_grad() = dtype == gx.dtype() ? gx : gx.AsType(dtype);
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([x1_tok = bb.RetainInput(0), x2_tok = bb.RetainInput(1), dtype = x2.dtype()](BackwardContext& bctx) {
                const Array& x1 = bctx.GetRetainedInput(x1_tok);
                const Array& x2 = bctx.GetRetainedInput(x2_tok);
                Array gx = -*bctx.output_grad() * FloorDivide(x1, x2);
                bctx.input_grad() = dtype == gx.dtype() ? std::move(gx) : gx.AsType(dtype);
            });
        }
        bb.Finalize();
    }
}

void ModASImpl(const Array& x1, Scalar x2, const Array& out) {
    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<ModASKernel>(x1, x2, out);
    }

    BackwardBuilder bb{"mod_array_scalar", x1, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x1_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& x1 = bctx.GetRetainedInput(x1_tok);
            const Array& gx = *bctx.output_grad();
            bctx.input_grad() = x1.dtype() == gx.dtype() ? gx : gx.AsType(x1.dtype());
        });
    }
    bb.Finalize();
}

void ModSAImpl(Scalar x1, const Array& x2, const Array& out) {
    {
        NoBackpropModeScope scope{};
        x2.device().backend().CallKernel<ModSAKernel>(x1, x2, out);
    }

    BackwardBuilder bb{"mod_scalar_array", x2, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([other = x1, x2_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& x2 = bctx.GetRetainedInput(x2_tok);
            bctx.input_grad() = -*bctx.output_grad() * FloorDivide(other, x2);
        });
    }
    bb.Finalize();
}

namespace internal {

void IMod(const Array& x1, const Array& x2) {
    CheckInplaceArithmeticDtypes(x1, x2);
    internal::BroadcastBinaryInplace(ModImpl, x1, x2);
}

void IMod(const Array& x1, Scalar x2) {
    CheckInplaceArithmeticDtypes(x1, x2);
    internal::BinaryInplace(&ModASImpl, x1, x2);
}

}  // namespace internal

Array Mod(const Array& x1, const Array& x2) { return internal::BroadcastBinary(&ModImpl, x1, x2, GetArithmeticResultDtype(x1, x2)); }

Array Mod(const Array& x1, Scalar x2) { return internal::Binary(&ModASImpl, x1, x2, GetArithmeticResultDtype(x1, x2)); }

Array Mod(Scalar x1, const Array& x2) { return internal::Binary(&ModSAImpl, x1, x2, GetArithmeticResultDtype(x1, x2)); }

}  // namespace chainerx
