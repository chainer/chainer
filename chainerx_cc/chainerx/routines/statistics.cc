#include "chainerx/routines/statistics.h"

#include "chainerx/array.h"
#include "chainerx/axes.h"

#include "chainerx/backprop_mode.h"
#include "chainerx/backward.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/arithmetic.h"
#include "chainerx/kernels/reduction.h"
#include "chainerx/kernels/statistics.h"
#include "chainerx/macro.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/type_util.h"

namespace chainerx {

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

Dtype PromoteInt2Float(Dtype dtype) { return GetKind(dtype) == DtypeKind::kFloat ? dtype : internal::GetDefaultDtype(DtypeKind::kFloat); }

Array Mean(const Array& a, const OptionalAxes& axis, bool keepdims) {
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, a.ndim());
    Dtype out_dtype = PromoteInt2Float(a.dtype());
    Array out = internal::EmptyReduced(a.shape(), out_dtype, sorted_axis, keepdims, a.device());
    Scalar n = internal::CountItemsAlongAxes(a.shape(), sorted_axis);

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<SumKernel>(a, sorted_axis, out);
        a.device().backend().CallKernel<DivideASKernel>(out, n, out);
    }

    BackwardBuilder bb{"mean", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([n, sorted_axis, in_shape = a.shape(), keepdims](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
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
    Dtype mean_out = PromoteInt2Float(a.dtype());
    // TODO(kshitij12345): remove once subtract allows mixed types.
    Array diff = a.AsType(mean_out, true) - Mean(a, sorted_axis, true);
    return Mean(diff * diff, sorted_axis, keepdims);
}

}  // namespace chainerx
