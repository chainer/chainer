#include "chainerx/routines/reduction.h"

#include <cmath>
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
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/reduction.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/statistics.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/shape.h"

namespace chainerx {

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

Array Softmax(const Array& x, const OptionalAxes& axis) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis.has_value() ? axis : OptionalAxes{1}, x.ndim());
    Array xmax = AMax(x_cast, sorted_axis, true);
    Array exps = Exp(x_cast - xmax);
    Array sums = Sum(exps, sorted_axis, true);
    return exps * Reciprocal(sums);
}

Array LogSumExp(const Array& x, const OptionalAxes& axis, bool keepdims) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, x.ndim());
    Array xmax = AMax(x_cast, sorted_axis, true);
    Array logs = Log(Sum(Exp(x_cast - xmax), sorted_axis, keepdims));
    return (keepdims ? xmax : Squeeze(xmax, axis)) + logs;
}

Array LogSoftmax(const Array& x, const OptionalAxes& axis) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return x_cast - LogSumExp(x_cast, axis.has_value() ? axis : OptionalAxes{1}, true);
}

}  // namespace chainerx
