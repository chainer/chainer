#include "chainerx/routines/math.h"

#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/enum.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/math.h"
#include "chainerx/macro.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {
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

Array Reciprocal(const Array& x) { return Scalar{1, GetKind(x.dtype())} / x; }

Array Sum(const Array& a, const OptionalAxes& axis, bool keepdims) {
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, a.ndim());

    // Decide the output dtype for integral input dtype.
    Dtype out_dtype{};
    switch (GetKind(a.dtype())) {
        case DtypeKind::kBool:
        case DtypeKind::kInt:  // fallthrough
            out_dtype = Dtype::kInt64;
            break;
        case DtypeKind::kUInt:
            out_dtype = Dtype::kInt64;  // TODO(niboshi): This should be kUInt64
            break;
        default:
            out_dtype = a.dtype();
    }

    Array out = internal::EmptyReduced(a.shape(), out_dtype, sorted_axis, keepdims, a.device());
    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<SumKernel>(a, sorted_axis, out);
    }

    BackwardBuilder bb{"sum", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([sorted_axis, in_shape = a.shape(), keepdims](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            CHAINERX_ASSERT(std::is_sorted(sorted_axis.begin(), sorted_axis.end()));

            if (!(in_shape.ndim() == 0 || sorted_axis.empty() || keepdims)) {
                Shape out_shape_broadcastable = gout.shape();
                for (auto axis : sorted_axis) {
                    out_shape_broadcastable.insert(out_shape_broadcastable.begin() + axis, 1);
                }
                bctx.input_grad() = gout.Reshape(out_shape_broadcastable).BroadcastTo(in_shape);
            } else {
                bctx.input_grad() = gout.BroadcastTo(in_shape);
            }
        });
    }
    bb.Finalize();
    return out;
}

Array AMax(const Array& a, const OptionalAxes& axis, bool keepdims) {
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, a.ndim());
    Array out = internal::EmptyReduced(a.shape(), a.dtype(), sorted_axis, keepdims, a.device());

    for (int8_t i : sorted_axis) {
        if (a.shape()[i] == 0) {
            throw DimensionError{"cannot compute the maximum along zero-sized axis"};
        }
    }

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<AMaxKernel>(a, sorted_axis, out);
    }

    BackwardBuilder bb{"amax", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        // a and out are used only for restoring the mask. We don't need graph nodes.
        bt.Define([sorted_axis, a = a.AsGradStopped(), out = out.AsGradStopped(), keepdims](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            CHAINERX_ASSERT(std::is_sorted(sorted_axis.begin(), sorted_axis.end()));

            Array reshaped_gout{};
            Array reshaped_out{};
            if (keepdims) {
                reshaped_gout = gout;
                reshaped_out = out;
            } else {
                // Add broadcastable dimensions to out and gout
                // for each one that was reduced in the forward operation
                Shape shape = internal::ReduceShape(a.shape(), sorted_axis, true);
                reshaped_gout = gout.Reshape(shape);
                reshaped_out = out.Reshape(shape);
            }

            // Compute the gradient
            // TODO(sonots): Use `where` if it becomes available.
            Array cond = (a == reshaped_out).AsType(gout.dtype(), false);
            bctx.input_grad() = reshaped_gout * cond;
        });
    }
    bb.Finalize();
    return out;
}

Array AMin(const Array& a, const OptionalAxes& axis, bool keepdims) {
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, a.ndim());
    Array out = internal::EmptyReduced(a.shape(), a.dtype(), sorted_axis, keepdims, a.device());

    for (int8_t i : sorted_axis) {
        if (a.shape()[i] == 0) {
            throw DimensionError{"cannot compute the minimum along zero-sized axis"};
        }
    }

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<AMinKernel>(a, sorted_axis, out);
    }

    BackwardBuilder bb{"amin", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        // a and out are used only for restoring the mask. We don't need graph nodes.
        bt.Define([sorted_axis, a = a.AsGradStopped(), out = out.AsGradStopped(), keepdims](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            CHAINERX_ASSERT(std::is_sorted(sorted_axis.begin(), sorted_axis.end()));

            Array reshaped_gout{};
            Array reshaped_out{};
            if (keepdims) {
                reshaped_gout = gout;
                reshaped_out = out;
            } else {
                // Add broadcastable dimensions to out and gout
                // for each one that was reduced in the forward operation
                Shape shape = internal::ReduceShape(a.shape(), sorted_axis, true);
                reshaped_gout = gout.Reshape(shape);
                reshaped_out = out.Reshape(shape);
            }

            // Compute the gradient
            // TODO(sonots): Use `where` if it becomes available.
            Array cond = (a == reshaped_out).AsType(gout.dtype(), false);
            bctx.input_grad() = reshaped_gout * cond;
        });
    }
    bb.Finalize();
    return out;
}

namespace {

// Calculates: x1 < x2 ? pos : neg
// Can only differentiate with respect to neg.
Array IfLessElse(const Array& x1, Scalar x2, Scalar pos, const Array& neg) {
    CheckArithmeticDtypes(GetKind(x1.dtype()), x2.kind(), false);
    Array out = Empty(x1.shape(), ResultType(pos, neg), x1.device());
    // TODO(niboshi): Create mask array and reuse in backprop.

    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<IfLessElseASSAKernel>(x1, x2, pos, neg, out);
    }

    BackwardBuilder bb{"if_less_else", neg, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x1 = x1.AsGradStopped(), x2](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            bctx.input_grad() = IfLessElse(x1, x2, Scalar{0, GetKind(gout.dtype())}, gout).AsType(x1.dtype(), false);
        });
    }
    bb.Finalize();

    return out;
}

Dtype GetMathResultDtype(Dtype dtype) {
    if (GetKind(dtype) == DtypeKind::kFloat) {
        return dtype;
    }
    return Dtype::kFloat32;  // TODO(niboshi): Default dtype
}

}  // namespace

namespace {

// Calculates: x1 > x2 ? pos : neg
// Can only differentiate with respect to neg.
Array IfGreaterElse(const Array& x1, Scalar x2, Scalar pos, const Array& neg) {
    CheckArithmeticDtypes(GetKind(x1.dtype()), x2.kind(), false);
    Array out = Empty(x1.shape(), ResultType(pos, neg), x1.device());
    // TODO(niboshi): Create mask array and reuse in backprop.

    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<IfGreaterElseASSAKernel>(x1, x2, pos, neg, out);
    }

    BackwardBuilder bb{"if_greater_else", neg, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x1 = x1.AsGradStopped(), x2](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            bctx.input_grad() = IfGreaterElse(x1, x2, Scalar{0, GetKind(gout.dtype())}, gout).AsType(x1.dtype(), false);
        });
    }
    bb.Finalize();

    return out;
}

}  // namespace

namespace {

void IfGreaterElseImpl(const Array& x1, const Array& x2, const Array& pos, const Array& neg, const Array& out) {
    CheckEqual(x1.shape(), x2.shape());
    {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<IfGreaterElseAAAAKernel>(x1, x2, pos, neg, out);
    }
    {
        BackwardBuilder bb{"if_greater_else", {pos, neg}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            // TODO(imanishi): Remove redundantly comparison x1 > x2 twice.
            Array mask = Greater(x1, x2);
            bt.Define([mask = std::move(mask), pos_dtype = pos.dtype()](BackwardContext& bctx) {
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = gout.AsType(pos_dtype, false) * mask;
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            // TODO(imanishi): Remove redundantly comparison x1 > x2 twice.
            Array not_mask = Less(x1, x2);
            bt.Define([not_mask = std::move(not_mask), neg_dtype = neg.dtype()](BackwardContext& bctx) {
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = gout.AsType(neg_dtype, false) * not_mask;
            });
        }
        bb.Finalize();
    }
}

}  // namespace

namespace {

void MinimumImpl(const Array& x1, const Array& x2, const Array& out) { IfGreaterElseImpl(x1, x2, x2, x1, out); }

}  // namespace

namespace {

void MaximumImpl(const Array& x1, const Array& x2, const Array& out) { IfGreaterElseImpl(x1, x2, x1, x2, out); }

}  // namespace

Array Maximum(const Array& x1, Scalar x2) {
    // TODO(niboshi): IfLessElse redundantly casts x1 twice.
    return IfLessElse(x1, x2, x2, x1);  // x1 < x2 ? x2 : x1
}

Array Maximum(Scalar x1, const Array& x2) { return Maximum(x2, x1); }

Array Maximum(const Array& x1, const Array& x2) {
    Dtype dtype = GetArithmeticResultDtype(x1, x2);
    return internal::BroadcastBinary(&MaximumImpl, x1, x2, dtype);  // x1 > x2 ? x1 : x2
}

Array Minimum(const Array& x1, Scalar x2) {
    // TODO(niboshi): IfGreaterElse redundantly casts x1 twice.
    return IfGreaterElse(x1, x2, x2, x1);  // x1 > x2 ? x2 : x1
}

Array Minimum(Scalar x1, const Array& x2) { return Minimum(x2, x1); }

Array Minimum(const Array& x1, const Array& x2) {
    Dtype dtype = GetArithmeticResultDtype(x1, x2);
    return internal::BroadcastBinary(&MinimumImpl, x1, x2, dtype);  // x1 > x2 ? x2 : x1
}

Array Exp(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<ExpKernel>(x, out);
    }

    BackwardBuilder bb{"exp", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([out_tok = bb.RetainOutput(0)](BackwardContext& bctx) {
            const Array& out = bctx.GetRetainedOutput(out_tok);
            bctx.input_grad() = *bctx.output_grad() * out;
        });
    }
    bb.Finalize();

    return out;
}

Array Log(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<LogKernel>(x, out);
    }

    BackwardBuilder bb{"log", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& x = bctx.GetRetainedInput(x_tok);
            bctx.input_grad() = *bctx.output_grad() / x;
        });
    }
    bb.Finalize();

    return out;
}

Array Log10(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<Log10Kernel>(x, out);
    }

    BackwardBuilder bb{"log10", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& x = bctx.GetRetainedInput(x_tok);
            bctx.input_grad() = *bctx.output_grad() / x * (1.0 / std::log(10));
        });
    }
    bb.Finalize();

    return out;
}

Array LogSumExp(const Array& x, const OptionalAxes& axis, bool keepdims) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, x.ndim());
    Array xmax = AMax(x_cast, sorted_axis, true);
    Array logs = Log(Sum(Exp(x_cast - xmax), sorted_axis, keepdims));
    return (keepdims ? xmax : Squeeze(xmax, axis)) + logs;
}

Array LogSoftmax(const Array& x, const OptionalAxes& axis) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return x_cast - LogSumExp(x_cast, axis.has_value() ? axis : OptionalAxes{1}, true);
}

Array Sigmoid(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return Reciprocal(1 + Exp(-x_cast));
}

Array Relu(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return Maximum(0, x_cast);
}

Array Softmax(const Array& x, const OptionalAxes& axis) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis.has_value() ? axis : OptionalAxes{1}, x.ndim());
    Array xmax = AMax(x_cast, sorted_axis, true);
    Array exps = Exp(x_cast - xmax);
    Array sums = Sum(exps, sorted_axis, true);
    return exps * Reciprocal(sums);
}

Array Square(const Array& x) {
    Array out = EmptyLike(x, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<SquareKernel>(x, out);
    }

    BackwardBuilder bb{"square", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& x = bctx.GetRetainedInput(x_tok);
            bctx.input_grad() = *bctx.output_grad() * (2 * x);
        });
    }
    bb.Finalize();

    return out;
}

Array SquaredDifference(const Array& x1, const Array& x2) { return Square(Subtract(x1, x2)); }

Array Sqrt(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<SqrtKernel>(x, out);
    }

    BackwardBuilder bb{"sqrt", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([out_tok = bb.RetainOutput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& out = bctx.GetRetainedOutput(out_tok);
            bctx.input_grad() = gout / (2 * out);
        });
    }
    bb.Finalize();

    return out;
}

Array Tanh(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<TanhKernel>(x, out);
    }

    BackwardBuilder bb{"tanh", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([out_tok = bb.RetainOutput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& out = bctx.GetRetainedOutput(out_tok);
            bctx.input_grad() = gout * (1 - out * out);
        });
    }
    bb.Finalize();

    return out;
}

Array Sin(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<SinKernel>(x, out);
    }

    BackwardBuilder bb{"sin", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([inp_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& inp = bctx.GetRetainedInput(inp_tok);
            bctx.input_grad() = gout * Cos(inp);
        });
    }
    bb.Finalize();

    return out;
}

Array Cos(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<CosKernel>(x, out);
    }

    BackwardBuilder bb{"cos", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([inp_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& inp = bctx.GetRetainedInput(inp_tok);
            bctx.input_grad() = gout * -Sin(inp);
        });
    }
    bb.Finalize();

    return out;
}

Array Absolute(const Array& x) {
    Array x_flip_1 = IfGreaterElse(x, 0.0, 0.0, -x);
    Array x_flip_2 = IfLessElse(x, 0.0, 0.0, x);

    Array out = x_flip_1 + x_flip_2;
    return out;
}

Array Tan(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<TanKernel>(x, out);
    }

    BackwardBuilder bb{"tan", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([inp_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& inp = bctx.GetRetainedInput(inp_tok);
            const Array& out = Cos(inp);
            bctx.input_grad() = gout / Square(out);
        });
    }
    bb.Finalize();

    return out;
}

Array Arcsin(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<ArcsinKernel>(x, out);
    }

    BackwardBuilder bb{"arcsin", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([inp_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& inp = bctx.GetRetainedInput(inp_tok);
            bctx.input_grad() = gout / Sqrt(1 - Square(inp));
        });
    }
    bb.Finalize();

    return out;
}

Array Arccos(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<ArccosKernel>(x, out);
    }

    BackwardBuilder bb{"arccos", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([inp_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& inp = bctx.GetRetainedInput(inp_tok);
            bctx.input_grad() = -gout / Sqrt(1 - Square(inp));
        });
    }
    bb.Finalize();

    return out;
}

Array Arctan(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<ArctanKernel>(x, out);
    }

    BackwardBuilder bb{"arctan", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([inp_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& inp = bctx.GetRetainedInput(inp_tok);
            bctx.input_grad() = gout / (1 + Square(inp));
        });
    }
    bb.Finalize();

    return out;
}

Array Arctan2(const Array& x1, const Array& x2) {
    Dtype out_dtype = GetMathResultDtype(GetArithmeticResultDtype(x1, x2));

    auto impl = [](const Array& x1, const Array& x2, Array& out) {
        CheckEqual(x1.shape(), x2.shape());

        {
            NoBackpropModeScope scope{};
            x1.device().backend().CallKernel<Arctan2Kernel>(x1, x2, out);
        }

        BackwardBuilder bb{"arctan2", {x1, x2}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([x1_tok = bb.RetainInput(0), x2_tok = bb.RetainInput(1)](BackwardContext& bctx) {
                const Array& gout = *bctx.output_grad();
                const Array& x1 = bctx.GetRetainedInput(x1_tok);
                const Array& x2 = bctx.GetRetainedInput(x2_tok);
                const Array& gin = gout * x2 / (Square(x1) + Square(x2));
                bctx.input_grad() = x1.dtype() != gin.dtype() ? gin.AsType(x1.dtype()) : gin;
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([x1_tok = bb.RetainInput(0), x2_tok = bb.RetainInput(1)](BackwardContext& bctx) {
                const Array& gout = *bctx.output_grad();
                const Array& x1 = bctx.GetRetainedInput(x1_tok);
                const Array& x2 = bctx.GetRetainedInput(x2_tok);
                const Array& gin = -gout * x1 / (Square(x1) + Square(x2));
                bctx.input_grad() = x2.dtype() != gin.dtype() ? gin.AsType(x2.dtype()) : gin;
            });
        }
        bb.Finalize();
    };

    return internal::BroadcastBinary(impl, x1, x2, out_dtype);
}

Array Sinh(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<SinhKernel>(x, out);
    }

    BackwardBuilder bb{"sinh", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([inp_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& inp = bctx.GetRetainedInput(inp_tok);
            bctx.input_grad() = gout * Cosh(inp);
        });
    }
    bb.Finalize();

    return out;
}

Array Cosh(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<CoshKernel>(x, out);
    }

    BackwardBuilder bb{"cosh", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([inp_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& inp = bctx.GetRetainedInput(inp_tok);
            bctx.input_grad() = gout * Sinh(inp);
        });
    }
    bb.Finalize();

    return out;
}

Array Arcsinh(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<ArcsinhKernel>(x, out);
    }

    BackwardBuilder bb{"arcsinh", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([inp_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& inp = bctx.GetRetainedInput(inp_tok);
            bctx.input_grad() = gout / Sqrt(1 + Square(inp));
        });
    }
    bb.Finalize();

    return out;
}

Array Arccosh(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<ArccoshKernel>(x, out);
    }

    BackwardBuilder bb{"arccosh", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([inp_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& inp = bctx.GetRetainedInput(inp_tok);
            bctx.input_grad() = gout / Sqrt(Square(inp) - 1);
        });
    }
    bb.Finalize();

    return out;
}

Array Ceil(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<CeilKernel>(x, out);
    }
    return out;
}

Array Floor(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<FloorKernel>(x, out);
    }
    return out;
}

Array IsNan(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<IsNanKernel>(x, out);
    }
    return out;
}

Array IsInf(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<IsInfKernel>(x, out);
    }
    return out;
}

Array IsFinite(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<IsFiniteKernel>(x, out);
    }
    return out;
}

template <typename Impl>
inline Array BitwiseImpl(Impl&& impl, const Array& x1, const Array& x2) {
    if (GetKind(x1.dtype()) == DtypeKind::kFloat || GetKind(x2.dtype()) == DtypeKind::kFloat) {
        throw DtypeError{"Bitwise operations don't support Float types"};
    }

    Dtype out_dtype = GetArithmeticResultDtype(x1, x2, true);
    return internal::BroadcastBinary(impl, x1, x2, out_dtype);
}

template <typename Kernel>
inline void ApplyBitwiseImpl(const Array& x1, const Array& x2, const Array& out) {
    NoBackpropModeScope scope;
    CheckEqual(x1.shape(), x2.shape());
    x1.device().backend().CallKernel<Kernel>(x1, x2, out);
}

template <typename Kernel>
inline void ApplyBitwiseASImpl(const Array& x1, Scalar x2, const Array& out) {
    NoBackpropModeScope scope;
    if (GetKind(x1.dtype()) == DtypeKind::kFloat || x2.kind() == DtypeKind::kFloat) {
        throw DtypeError{"Bitwise operations don't support Float types"};
    }
    x1.device().backend().CallKernel<Kernel>(x1, x2, out);
}

namespace internal {

template <typename Impl>
inline void IBitwiseImpl(Impl&& impl, const Array& x1, const Array& x2) {
    if (GetKind(x1.dtype()) == DtypeKind::kFloat || GetKind(x2.dtype()) == DtypeKind::kFloat) {
        throw DtypeError{"Bitwise operations don't support Float types"};
    }

    CheckInplaceArithmeticDtypes(x1, x2, true);
    internal::BroadcastBinaryInPlace(impl, x1, x2);
}

void IBitwiseAnd(const Array& x1, const Array& x2) { IBitwiseImpl(ApplyBitwiseImpl<BitwiseAndKernel>, x1, x2); }

void IBitwiseAnd(const Array& x1, Scalar x2) {
    CheckInplaceArithmeticDtypes(x1, x2, true);
    internal::BinaryInPlace(ApplyBitwiseASImpl<BitwiseAndASKernel>, x1, x2);
}

void IBitwiseOr(const Array& x1, const Array& x2) { IBitwiseImpl(ApplyBitwiseImpl<BitwiseOrKernel>, x1, x2); }

void IBitwiseOr(const Array& x1, Scalar x2) {
    CheckInplaceArithmeticDtypes(x1, x2, true);
    internal::BinaryInPlace(ApplyBitwiseASImpl<BitwiseOrASKernel>, x1, x2);
}

void IBitwiseXor(const Array& x1, const Array& x2) { IBitwiseImpl(ApplyBitwiseImpl<BitwiseXorKernel>, x1, x2); }

void IBitwiseXor(const Array& x1, Scalar x2) {
    CheckInplaceArithmeticDtypes(x1, x2, true);
    internal::BinaryInPlace(ApplyBitwiseASImpl<BitwiseXorASKernel>, x1, x2);
}

}  // namespace internal

Array BitwiseAnd(const Array& x1, const Array& x2) { return BitwiseImpl(ApplyBitwiseImpl<BitwiseAndKernel>, x1, x2); }

Array BitwiseAnd(const Array& x1, Scalar x2) {
    return internal::Binary(ApplyBitwiseASImpl<BitwiseAndASKernel>, x1, x2, GetArithmeticResultDtype(x1, x2, true));
}

Array BitwiseAnd(Scalar x1, const Array& x2) { return BitwiseAnd(x2, x1); }

Array BitwiseOr(const Array& x1, const Array& x2) { return BitwiseImpl(ApplyBitwiseImpl<BitwiseOrKernel>, x1, x2); }

Array BitwiseOr(const Array& x1, Scalar x2) {
    return internal::Binary(ApplyBitwiseASImpl<BitwiseOrASKernel>, x1, x2, GetArithmeticResultDtype(x1, x2, true));
}

Array BitwiseOr(Scalar x1, const Array& x2) { return BitwiseOr(x2, x1); }

Array BitwiseXor(const Array& x1, const Array& x2) { return BitwiseImpl(ApplyBitwiseImpl<BitwiseXorKernel>, x1, x2); }

Array BitwiseXor(const Array& x1, Scalar x2) {
    return internal::Binary(ApplyBitwiseASImpl<BitwiseXorASKernel>, x1, x2, GetArithmeticResultDtype(x1, x2, true));
}

Array BitwiseXor(Scalar x1, const Array& x2) { return BitwiseXor(x2, x1); }

}  // namespace chainerx
