#include "chainerx/routines/statistics.h"

#include "chainerx/array.h"
#include "chainerx/axes.h"

#include "chainerx/backprop_mode.h"
#include "chainerx/backward.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"

namespace chainerx {

Array Mean(const Array& a, const OptionalAxes& axis, bool keepdims) {
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, a.ndim());
    Array out = internal::EmptyReduced(a.shape(), a.dtype(), sorted_axis, keepdims, a.device());
    Scalar n = internal::CountItemsAlongAxes(a.shape(), sorted_axis);

    {
        NoBackpropModeScope scope{};
        a.device().Sum(a, sorted_axis, out);
        a.device().DivideAS(out, n, out);
    }

    BackwardBuilder bb{"mean", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([n, sorted_axis, in_shape = a.shape(), keepdims](BackwardContext& bctx) {
            const Array& gout = bctx.output_grad();
            CHAINERX_ASSERT(std::is_sorted(sorted_axis.begin(), sorted_axis.end()));

            if (!(in_shape.ndim() == 0 || sorted_axis.empty() || keepdims)) {
                Shape out_shape_broadcastable = gout.shape();
                for (auto axis : sorted_axis) {
                    out_shape_broadcastable.insert(out_shape_broadcastable.begin() + axis, 1);
                }
                bctx.input_grad() = gout.Reshape(out_shape_broadcastable).BroadcastTo(in_shape) / n;
            } else {
                bctx.input_grad() = gout.BroadcastTo(in_shape) / n;
            }
        });
    }
    bb.Finalize();

    return out;
}

Array Var(const Array& a, const OptionalAxes& axis, bool keepdims) {
    // TODO(hvy): Consider allowing device implementations.
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, a.ndim());
    Array diff = a - Mean(a, sorted_axis, true);
    return Mean(diff * diff, sorted_axis, keepdims);
}

}  // namespace chainerx
