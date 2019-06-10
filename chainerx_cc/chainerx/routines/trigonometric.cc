#include "chainerx/routines/trigonometric.h"

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/trigonometric.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/shape.h"

namespace chainerx {

Array Sin(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
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
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
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

Array Tan(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
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
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
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
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
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
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
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
    Dtype out_dtype = internal::GetMathResultDtype(ResultType(x1, x2));

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

}  // namespace chainerx
