#include "chainerx/routines/math.h"

#include <cstdint>
#include <numeric>
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
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/routines_util.h"
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

// Called from Add, Subtract, Multiply, Divide, etc. to handle broadcasting.
template <typename Impl>
Array BroadcastBinary(Impl&& impl, const Array& x1, const Array& x2) {
    auto func = [&impl](const Array& x1, const Array& x2) -> Array {
        // TODO(hvy): Use type promotion for output.
        Array out = EmptyLike(x1, x1.device());
        impl(x1, x2, out);
        return out;
    };

    if (x1.shape() == x2.shape()) {
        return func(x1, x2);
    }
    Shape result_shape = internal::BroadcastShapes(x1.shape(), x2.shape());
    if (x1.shape() == result_shape) {
        return func(x1, x2.BroadcastTo(result_shape));
    }
    if (x2.shape() == result_shape) {
        return func(x1.BroadcastTo(result_shape), x2);
    }
    return func(x1.BroadcastTo(result_shape), x2.BroadcastTo(result_shape));
}

// Called from IAdd, ISubtract, IMultiply, IDivide, etc. to handle broadcasting.
template <typename Impl>
void BroadcastBinaryInPlace(Impl&& impl, const Array& x1, const Array& x2) {
    internal::CheckNoUnsafeInplace(x1, {x1, x2});
    if (x1.shape() == x2.shape()) {
        impl(x1, x2, x1);
    } else {
        impl(x1, x2.BroadcastTo(x1.shape()), x1);
    }
}

template <typename Impl>
Array Binary(Impl&& impl, const Array& x1, Scalar x2) {
    // TODO(hvy): Use type promotion for output.
    Array out = EmptyLike(x1, x1.device());
    impl(x1, x2, out);
    return out;
}

template <typename Impl>
void BinaryInPlace(Impl&& impl, const Array& x1, Scalar x2) {
    internal::CheckNoUnsafeInplace(x1, {x1});
    impl(x1, x2, x1);
}

void AddImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(sonots): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    {
        NoBackpropModeScope scope{};
        x1.device().Add(x1, x2, out);
    }

    {
        BackwardBuilder bb{"add", {x1, x2}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
        }
        bb.Finalize();
    }
}

void AddASImpl(const Array& x1, Scalar x2, const Array& out) {
    // TODO(hvy): dtype conversion

    {
        NoBackpropModeScope scope{};
        x1.device().AddAS(x1, x2, out);
    }

    BackwardBuilder bb{"add_scalar", x1, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
    }
    bb.Finalize();
}

}  // namespace

namespace internal {

void IAdd(const Array& x1, const Array& x2) { BroadcastBinaryInPlace(&AddImpl, x1, x2); }

void IAdd(const Array& x1, Scalar x2) { BinaryInPlace(&AddASImpl, x1, x2); }

}  // namespace internal

Array Add(const Array& x1, const Array& x2) { return BroadcastBinary(&AddImpl, x1, x2); }

Array Add(const Array& x1, Scalar x2) { return Binary(&AddASImpl, x1, x2); }

Array Add(Scalar x1, const Array& x2) { return Add(x2, x1); }

namespace {

void SubtractImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(niboshi): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());
    if (x1.dtype() == Dtype::kBool) {
        throw DtypeError{"Cannot subtract from a boolean array."};
    }

    {
        NoBackpropModeScope scope{};
        x1.device().Subtract(x1, x2, out);
    }

    {
        BackwardBuilder bb{"subtract", {x1, x2}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([](BackwardContext& bctx) { bctx.input_grad() = -*bctx.output_grad(); });
        }
        bb.Finalize();
    }
}

void SubtractASImpl(const Array& x1, Scalar x2, const Array& out) {
    // TODO(hvy): dtype conversion
    if (x1.dtype() == Dtype::kBool) {
        throw DtypeError{"Cannot subtract from a boolean array."};
    }

    {
        NoBackpropModeScope scope{};
        x1.device().SubtractAS(x1, x2, out);
    }

    BackwardBuilder bb{"subtract_scalar", x1, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
    }
    bb.Finalize();
}

}  // namespace

namespace internal {

void ISubtract(const Array& x1, const Array& x2) { BroadcastBinaryInPlace(&SubtractImpl, x1, x2); }

void ISubtract(const Array& x1, Scalar x2) { BinaryInPlace(&SubtractASImpl, x1, x2); }

}  // namespace internal

Array Subtract(const Array& x1, const Array& x2) { return BroadcastBinary(&SubtractImpl, x1, x2); }

Array Subtract(const Array& x1, Scalar x2) { return Binary(&SubtractASImpl, x1, x2); }

Array Subtract(Scalar x1, const Array& x2) { return Add(-x2, x1); }

namespace {

void MultiplyImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(sonots): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    {
        NoBackpropModeScope scope{};
        x1.device().Multiply(x1, x2, out);
    }

    {
        BackwardBuilder bb{"multiply", {x1, x2}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([x2_tok = bb.RetainInput(1)](BackwardContext& bctx) {
                const Array& x2 = bctx.GetRetainedInput(x2_tok);
                bctx.input_grad() = *bctx.output_grad() * x2;
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([x1_tok = bb.RetainInput(0)](BackwardContext& bctx) {
                const Array& x1 = bctx.GetRetainedInput(x1_tok);
                bctx.input_grad() = *bctx.output_grad() * x1;
            });
        }
        bb.Finalize();
    }
}

void MultiplyASImpl(const Array& x1, Scalar x2, const Array& out) {
    // TODO(hvy): dtype conversion

    {
        NoBackpropModeScope scope{};
        x1.device().MultiplyAS(x1, x2, out);
    }

    BackwardBuilder bb{"multiply_scalar", x1, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x2](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad() * x2; });
    }
    bb.Finalize();
}

}  // namespace

namespace internal {

void IMultiply(const Array& x1, const Array& x2) { BroadcastBinaryInPlace(&MultiplyImpl, x1, x2); }

void IMultiply(const Array& x1, Scalar x2) { BinaryInPlace(&MultiplyASImpl, x1, x2); }

}  // namespace internal

Array Multiply(const Array& x1, const Array& x2) { return BroadcastBinary(&MultiplyImpl, x1, x2); }

Array Multiply(const Array& x1, Scalar x2) { return Binary(&MultiplyASImpl, x1, x2); }

Array Multiply(Scalar x1, const Array& x2) { return Multiply(x2, x1); }

namespace {

void FloorDivideImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(imanishi): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    {
        NoBackpropModeScope scope{};
        x1.device().FloorDivide(x1, x2, out);
    }
}

void FloorDivideASImpl(const Array& x1, Scalar x2, const Array& out) {
    // TODO(imanishi): dtype conversion

    {
        NoBackpropModeScope scope{};
        x1.device().FloorDivideAS(x1, x2, out);
    }
}

void DivideImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(niboshi): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    {
        NoBackpropModeScope scope{};
        x1.device().Divide(x1, x2, out);
    }

    {
        BackwardBuilder bb{"divide", {x1, x2}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([x2_tok = bb.RetainInput(1)](BackwardContext& bctx) {
                const Array& x2 = bctx.GetRetainedInput(x2_tok);
                bctx.input_grad() = *bctx.output_grad() / x2;
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([x1_tok = bb.RetainInput(0), x2_tok = bb.RetainInput(1)](BackwardContext& bctx) {
                const Array& x1 = bctx.GetRetainedInput(x1_tok);
                const Array& x2 = bctx.GetRetainedInput(x2_tok);
                bctx.input_grad() = -*bctx.output_grad() * x1 / (x2 * x2);
            });
        }
        bb.Finalize();
    }
}

void DivideASImpl(const Array& x1, Scalar x2, const Array& out) {
    // TODO(hvy): dtype conversion

    {
        NoBackpropModeScope scope{};
        x1.device().DivideAS(x1, x2, out);
    }

    BackwardBuilder bb{"divide_scalar", x1, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([other = x2](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad() / other; });
    }
    bb.Finalize();
}

}  // namespace

namespace internal {

void IFloorDivide(const Array& x1, const Array& x2) { BroadcastBinaryInPlace(&FloorDivideImpl, x1, x2); }

void IFloorDivide(const Array& x1, Scalar x2) { BinaryInPlace(&FloorDivideASImpl, x1, x2); }

void ITrueDivide(const Array& x1, const Array& x2) {
    if (GetKind(x1.dtype()) != DtypeKind::kFloat) {
        throw DtypeError{"Integer inplace-division is not supported."};
    }
    BroadcastBinaryInPlace(&DivideImpl, x1, x2);
}

void ITrueDivide(const Array& x1, Scalar x2) {
    if (GetKind(x1.dtype()) != DtypeKind::kFloat) {
        throw DtypeError{"Integer inplace-division is not supported."};
    }
    BinaryInPlace(&DivideASImpl, x1, x2);
}

void IDivide(const Array& x1, const Array& x2) { ITrueDivide(x1, x2); }

void IDivide(const Array& x1, Scalar x2) { ITrueDivide(x1, x2); }

}  // namespace internal

Array FloorDivide(const Array& x1, const Array& x2) { return BroadcastBinary(&FloorDivideImpl, x1, x2); }

Array FloorDivide(const Array& x1, Scalar x2) { return Binary(&FloorDivideASImpl, x1, x2); }

Array FloorDivide(Scalar /*x1*/, const Array& /*x2*/) { throw NotImplementedError{"Scalar / Array division is not yet supported."}; }

Array TrueDivide(const Array& x1, const Array& x2) {
    if (GetKind(x1.dtype()) == DtypeKind::kFloat) {
        return BroadcastBinary(&DivideImpl, x1, x2);
    }
    CheckEqual(x1.dtype(), x2.dtype());
    return BroadcastBinary(&DivideImpl, x1.AsType(Dtype::kFloat64), x2.AsType(Dtype::kFloat64));
}

Array TrueDivide(const Array& x1, Scalar x2) {
    if (GetKind(x1.dtype()) == DtypeKind::kFloat) {
        return Binary(&DivideASImpl, x1, x2);
    }
    return Binary(&DivideASImpl, x1.AsType(Dtype::kFloat64), Scalar{static_cast<double>(x2)});
}

Array TrueDivide(Scalar /*x1*/, const Array& /*x2*/) { throw NotImplementedError{"Scalar / Array division is not yet supported."}; }

Array Divide(const Array& x1, const Array& x2) { return TrueDivide(x1, x2); }

Array Divide(const Array& x1, Scalar x2) { return TrueDivide(x1, x2); }

Array Divide(Scalar x1, const Array& x2) { return TrueDivide(x1, x2); }

Array Reciprocal(const Array& x) {
    // TODO(hvy): Optimize the implementation using e.g. 1 / x.
    return OnesLike(x, x.device()) / x;
}

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
        a.device().Sum(a, sorted_axis, out);
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
        a.device().AMax(a, sorted_axis, out);
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

namespace {

// Calculates: x1 < x2 ? pos : neg
// Can only differentiate with respect to neg.
Array IfLessElse(const Array& x1, Scalar x2, Scalar pos, const Array& neg) {
    Array out = EmptyLike(x1, x1.device());

    {
        NoBackpropModeScope scope{};
        x1.device().IfLessElseASSA(x1, x2, pos, neg, out);
    }

    BackwardBuilder bb{"if_less_else", neg, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x1 = x1.AsGradStopped(), x2](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            bctx.input_grad() = IfLessElse(x1, x2, Scalar{0, GetKind(gout.dtype())}, gout);
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
    Array out = EmptyLike(x1, x1.device());

    {
        NoBackpropModeScope scope{};
        x1.device().IfGreaterElseASSA(x1, x2, pos, neg, out);
    }

    BackwardBuilder bb{"if_greater_else", neg, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x1 = x1.AsGradStopped(), x2](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            bctx.input_grad() = IfGreaterElse(x1, x2, Scalar{0, GetKind(gout.dtype())}, gout);
        });
    }
    bb.Finalize();

    return out;
}

}  // namespace

Array Maximum(const Array& x1, Scalar x2) {
    return IfLessElse(x1, x2, x2, x1);  // x1 < x2 ? x2 : x1
}

Array Maximum(Scalar x1, const Array& x2) { return Maximum(x2, x1); }

Array Minimum(const Array& x1, Scalar x2) {
    return IfGreaterElse(x1, x2, x2, x1);  // x1 > x2 ? x2 : x1
}

Array Minimum(Scalar x1, const Array& x2) { return Minimum(x2, x1); }

Array Exp(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().Exp(x, out);
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
        x.device().Log(x, out);
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

Array LogSumExp(const Array& x, const OptionalAxes& axis, bool keepdims) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, x.ndim());
    Array xmax = AMax(x_cast, sorted_axis, true);
    Array logs = Log(Sum(Exp(x_cast - xmax), sorted_axis, keepdims));

    // TODO(imanishi): Avoid unnecessary cast here when `chainerx::Add` supports mixed dtypes.
    return (keepdims ? xmax : Squeeze(xmax, axis)) + logs;
}

Array LogSoftmax(const Array& x, const OptionalAxes& axis) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);

    // TODO(imanishi): Avoid unnecessary cast here when `chainerx::Subtract` supports mixed dtypes.
    return x_cast - LogSumExp(x_cast, axis.has_value() ? axis : OptionalAxes{1}, true);
}

Array Sqrt(const Array& x) {
    Dtype dtype = GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().Sqrt(x, out);
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
        x.device().Tanh(x, out);
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

Array IsNan(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().IsNan(x, out);
    }
    return out;
}

Array IsInf(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().IsInf(x, out);
    }
    return out;
}

}  // namespace chainerx
