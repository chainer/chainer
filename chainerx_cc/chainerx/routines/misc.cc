#include "chainerx/routines/misc.h"

#include <utility>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"

namespace chainerx {

namespace {

void CheckComparisonDtypes(DtypeKind kind1, DtypeKind kind2) {
    if ((kind1 == DtypeKind::kBool) != (kind2 == DtypeKind::kBool)) {
        throw DtypeError{"Comparison of bool and non-bool dtypes is not supported."};
    }
}

void CheckComparisonDtypes(const Array& x1, const Array& x2) { return CheckComparisonDtypes(GetKind(x1.dtype()), GetKind(x2.dtype())); }

void CheckComparisonDtypes(const Array& x1, Scalar x2) { return CheckComparisonDtypes(GetKind(x1.dtype()), x2.kind()); }

// Calculates: x1 < x2 ? pos : neg
// Can only differentiate with respect to neg.
Array IfLessElse(const Array& x1, Scalar x2, Scalar pos, const Array& neg) {
    CheckComparisonDtypes(x1, x2);
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

// Calculates: x1 > x2 ? pos : neg
// Can only differentiate with respect to neg.
Array IfGreaterElse(const Array& x1, Scalar x2, Scalar pos, const Array& neg) {
    CheckComparisonDtypes(x1, x2);
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

void IfGreaterElseImpl(const Array& x1, const Array& x2, const Array& pos, const Array& neg, const Array& out) {
    CheckComparisonDtypes(x1, x2);
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

void MinimumImpl(const Array& x1, const Array& x2, const Array& out) { IfGreaterElseImpl(x1, x2, x2, x1, out); }

void MaximumImpl(const Array& x1, const Array& x2, const Array& out) { IfGreaterElseImpl(x1, x2, x1, x2, out); }

}  // namespace

Array Sqrt(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
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

Array Square(const Array& x) {
    if (x.dtype() == Dtype::kBool) {
        throw DtypeError{"Square operation don't support Boolean type"};
    }
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

Array SquaredDifference(const Array& x1, const Array& x2) { return Square(x1 - x2); }

Array Absolute(const Array& x) {
    Array x_flip_1 = IfGreaterElse(x, 0.0, 0.0, -x);
    Array x_flip_2 = IfLessElse(x, 0.0, 0.0, x);

    Array out = x_flip_1 + x_flip_2;
    return out;
}

Array Fabs(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<FabsKernel>(x, out);
    }

    BackwardBuilder bb{"fabs", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([inp_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& inp = bctx.GetRetainedInput(inp_tok);
            bctx.input_grad() = gout * Sign(inp);
        });
    }
    bb.Finalize();

    return out;
}

Array Sign(const Array& x) {
    Array out = Empty(x.shape(), x.dtype(), x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<SignKernel>(x, out);
    }
    return out;
}

Array Maximum(const Array& x1, Scalar x2) {
    if (x1.dtype() == Dtype::kBool && x2.kind() == DtypeKind::kBool) {
        return LogicalOr(x1, x2);
    }
    // TODO(niboshi): IfLessElse redundantly casts x1 twice.
    return IfLessElse(x1, x2, x2, x1);  // x1 < x2 ? x2 : x1
}

Array Maximum(Scalar x1, const Array& x2) { return Maximum(x2, x1); }

Array Maximum(const Array& x1, const Array& x2) {
    if (x1.dtype() == Dtype::kBool && x2.dtype() == Dtype::kBool) {
        return LogicalOr(x1, x2);
    }
    Dtype dtype = ResultType(x1, x2);
    return internal::BroadcastBinary(&MaximumImpl, x1, x2, dtype);  // x1 > x2 ? x1 : x2
}

Array Minimum(const Array& x1, Scalar x2) {
    if (x1.dtype() == Dtype::kBool && x2.kind() == DtypeKind::kBool) {
        return LogicalAnd(x1, x2);
    }
    // TODO(niboshi): IfGreaterElse redundantly casts x1 twice.
    return IfGreaterElse(x1, x2, x2, x1);  // x1 > x2 ? x2 : x1
}

Array Minimum(Scalar x1, const Array& x2) { return Minimum(x2, x1); }

Array Minimum(const Array& x1, const Array& x2) {
    if (x1.dtype() == Dtype::kBool && x2.dtype() == Dtype::kBool) {
        return LogicalAnd(x1, x2);
    }
    Dtype dtype = ResultType(x1, x2);
    return internal::BroadcastBinary(&MinimumImpl, x1, x2, dtype);  // x1 > x2 ? x2 : x1
}

}  // namespace chainerx
