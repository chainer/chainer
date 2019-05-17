#include "chainerx/routines/hyperbolic.h"

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/hyperbolic.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace {

Dtype GetMathResultDtype(Dtype dtype) {
    if (GetKind(dtype) == DtypeKind::kFloat) {
        return dtype;
    }
    return Dtype::kFloat32;  // TODO(niboshi): Default dtype
}

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

}  // namespace chainerx
