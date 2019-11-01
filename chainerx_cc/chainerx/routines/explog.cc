#include "chainerx/routines/explog.h"

#include <cmath>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/constant.h"
#include "chainerx/dtype.h"
#include "chainerx/enum.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/explog.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/type_util.h"

namespace chainerx {

Array Erf(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<ErfKernel>(x, out);
    }

    BackwardBuilder bb{"erf", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& x = bctx.GetRetainedInput(x_tok);
            bctx.input_grad() = *bctx.output_grad() * 2.0 / static_cast<double>(std::sqrt(kPi)) * Exp(-Square(x));
        });
    }
    bb.Finalize();

    return out;
}

Array Exp(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
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

Array Expm1(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<Expm1Kernel>(x, out);
    }

    BackwardBuilder bb{"expm1", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([out_tok = bb.RetainOutput(0)](BackwardContext& bctx) {
            const Array& out = bctx.GetRetainedOutput(out_tok);
            bctx.input_grad() = *bctx.output_grad() * (out + 1);
        });
    }
    bb.Finalize();

    return out;
}

Array Exp2(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<Exp2Kernel>(x, out);
    }

    BackwardBuilder bb{"exp2", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([out_tok = bb.RetainOutput(0)](BackwardContext& bctx) {
            const Array& out = bctx.GetRetainedOutput(out_tok);
            bctx.input_grad() = *bctx.output_grad() * out * std::log(2.0);
        });
    }
    bb.Finalize();

    return out;
}

Array Log(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
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
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
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

Array Log2(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<Log2Kernel>(x, out);
    }

    BackwardBuilder bb{"log2", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& x = bctx.GetRetainedInput(x_tok);
            bctx.input_grad() = *bctx.output_grad() / x * (1.0 / std::log(2.0));
        });
    }
    bb.Finalize();

    return out;
}

Array Log1p(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<Log1pKernel>(x, out);
    }

    BackwardBuilder bb{"log1p", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([x_tok = bb.RetainInput(0)](BackwardContext& bctx) {
            const Array& x = bctx.GetRetainedInput(x_tok);
            bctx.input_grad() = *bctx.output_grad() / (1.0 + x);
        });
    }
    bb.Finalize();

    return out;
}

}  // namespace chainerx
