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

    if (!x.IsConstant() || !gamma.IsConstant() || !beta.IsConstant()) {
        BackwardBuilder bb{"batch_norm", {out}};
        bb.Define({x, gamma, beta}, [ fb = std::move(fb), x, gamma = gamma_keepdims, eps, sorted_axis ](BackwardContext & bctx) {
            const Array& gout = bctx.output_grad();
            auto ginputs = fb->Backward(x, gamma, gout, eps, sorted_axis);
            static_assert(sizeof(ginputs) / sizeof(ginputs[0]) == 3, "Backward of BatchNorm is expecting 3 gradients.");
            const Array& gx = ginputs[0];
            const Array& ggamma = ginputs[1];
            const Array& gbeta = ginputs[2];
            assert(gx.IsConstant());
            assert(ggamma.IsConstant());
            assert(gbeta.IsConstant());

            Array x_cut = bctx.Cut(x);
            Array gamma_cut = bctx.Cut(gamma);

            if (bctx.next_required() && (!x_cut.IsConstant() || !gamma_cut.IsConstant() || !gout.IsConstant())) {
                BackwardBuilder bb2{"batch_norm_backward", {gx, ggamma, gbeta}};
                bb2.Define({x_cut, gamma_cut, gout}, [fb](BackwardContext& bctx2) {
                    const Array& g2x = bctx2.output_grad(0);
                    const Array& g2gamma = bctx2.output_grad(1);
                    const Array& g2beta = bctx2.output_grad(2);
                    auto ginputs2 = fb->DoubleBackward(g2x, g2gamma, g2beta);
                    // TODO(niboshi): Make it further backproppable
                    static_assert(sizeof(ginputs2) / sizeof(ginputs2[0]) == 3, "Double backward of BatchNorm is expecting 3 gradients.");
                    // TODO(niboshi): Assign at once
                    bctx2.input_grad(0) = ginputs2[0];  // ggx
                    bctx2.input_grad(1) = ginputs2[1];  // gggamma
                    bctx2.input_grad(2) = ginputs2[2];  // ggout
                });
            }

            // TODO(niboshi): Assign at once
            bctx.input_grad(0) = gx;
            bctx.input_grad(1) = ggamma;
            bctx.input_grad(2) = gbeta;
        });
    }

    return out;
}

}  // namespace xchainer
