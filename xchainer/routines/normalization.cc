#include "xchainer/routines/normalization.h"

#include <cstdint>
#include <memory>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/backward.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {

Array BatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        Scalar eps,
        Scalar decay,
        const OptionalAxes& axis) {
    Dtype dtype = x.dtype();
    CheckEqual(dtype, gamma.dtype());
    CheckEqual(dtype, beta.dtype());
    CheckEqual(dtype, running_mean.dtype());
    CheckEqual(dtype, running_var.dtype());

    Axes sorted_axis = axis.has_value() ? internal::GetSortedAxes(*axis, x.ndim()) : Axes{0};

    Shape reduced_shape = internal::ReduceShape(x.shape(), sorted_axis, true);
    int64_t reduced_size = reduced_shape.GetTotalSize();

    if (gamma.GetTotalSize() != reduced_size) {
        throw DimensionError{
                "Gamma must have the same size as the reduced input. Actual: ", gamma.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (beta.GetTotalSize() != reduced_size) {
        throw DimensionError{
                "Beta must have the same size as the reduced input. Actual: ", beta.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (running_mean.GetTotalSize() != reduced_size) {
        throw DimensionError{"Running mean must have the same size as the reduced input. Actual: ",
                             running_mean.GetTotalSize(),
                             ". Expected: ",
                             reduced_size,
                             "."};
    }
    if (running_var.GetTotalSize() != reduced_size) {
        throw DimensionError{"Running variance must have the same size as the reduced input. Actual: ",
                             running_var.GetTotalSize(),
                             ". Expected: ",
                             reduced_size,
                             "."};
    }

    Array gamma_keepdims = gamma.shape() == reduced_shape ? gamma : gamma.Reshape(reduced_shape);
    Array beta_keepdims = beta.shape() == reduced_shape ? beta : beta.Reshape(reduced_shape);

    std::shared_ptr<BatchNormForwardBackward> fb = x.device().GetBatchNormForwardBackward();
    Array running_mean_view = running_mean.Reshape(reduced_shape);
    Array running_var_view = running_var.Reshape(reduced_shape);
    assert(running_mean_view.data() == running_mean.data());  // No copy should occur
    assert(running_var_view.data() == running_var.data());

    Array out = fb->Forward(x, gamma_keepdims, beta_keepdims, running_mean_view, running_var_view, eps, decay, sorted_axis);

    {
        BackwardBuilder bb{"batch_norm", {out}};
        if (!x.IsConstant() || !gamma.IsConstant() || !beta.IsConstant()) {
            bb.Define(
                    {x, gamma, beta},
                    [ fb = std::move(fb), x = x.AsConstant(), gamma_keepdims = gamma_keepdims.AsConstant(), eps, sorted_axis ](
                            BackwardContext & bctx) {
                        const Array& gout = bctx.output_grad();
                        auto ginputs = fb->Backward(x, gamma_keepdims, gout, eps, sorted_axis);
                        // TODO(niboshi): Implement double backward

                        // TODO(niboshi): Implement a convenient function in BackwardContext to move arrays from a container
                        for (size_t i = 0; i < ginputs.size(); ++i) {
                            bctx.input_grad(i) = std::move(ginputs[i]);
                        }
                    });
        }
    }

    return out;
}

}  // namespace xchainer
