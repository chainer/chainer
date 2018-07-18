#include "xchainer/routines/normalization.h"

#include <cstdint>
#include <memory>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/backprop_mode.h"
#include "xchainer/backward_builder.h"
#include "xchainer/backward_context.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/routines/routines_util.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace {

struct PreprocessBatchNormResult {
    // Arrays are reshaped if necessary
    Array gamma;
    Array beta;
    Array mean;
    Array var;
    Axes sorted_axis;
};

// Reshapes the array. If the shape is unchanged, an array with identical array body is returned. Note that xchainer::Reshape() returns
// a view with different array body if the shape is unchanged.
Array ReshapeOrIdentity(const Array& a, const Shape& shape) {
    if (a.shape() == shape) {
        return a;
    }
    return a.Reshape(shape);
}

// Reshapes the input arrays (except x) as needed.
// Sorted axes is also returned.
PreprocessBatchNormResult PreprocessBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, const OptionalAxes& axis) {
    Dtype dtype = x.dtype();
    CheckEqual(dtype, gamma.dtype());
    CheckEqual(dtype, beta.dtype());
    CheckEqual(dtype, mean.dtype());
    CheckEqual(dtype, var.dtype());

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
    if (mean.GetTotalSize() != reduced_size) {
        throw DimensionError{
                "Mean must have the same size as the reduced input. Actual: ", mean.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (var.GetTotalSize() != reduced_size) {
        throw DimensionError{
                "Variance must have the same size as the reduced input. Actual: ", var.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }

    Array gamma_reshaped = ReshapeOrIdentity(gamma, reduced_shape);
    Array beta_reshaped = ReshapeOrIdentity(beta, reduced_shape);
    Array mean_reshaped = ReshapeOrIdentity(mean, reduced_shape);
    Array var_reshaped = ReshapeOrIdentity(var, reduced_shape);
    assert(gamma_reshaped.data() == gamma.data());  // No data copy should occur
    assert(beta_reshaped.data() == beta.data());
    assert(mean_reshaped.data() == mean.data());
    assert(var_reshaped.data() == var.data());

    return {std::move(gamma_reshaped), std::move(beta_reshaped), std::move(mean_reshaped), std::move(var_reshaped), sorted_axis};
}

}  // namespace

Array BatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        Scalar eps,
        Scalar decay,
        const OptionalAxes& axis) {
    PreprocessBatchNormResult result = PreprocessBatchNorm(x, gamma, beta, running_mean, running_var, axis);
    std::shared_ptr<BatchNormForwardBackward> fb =
            x.device().GetBatchNormForwardBackward(result.mean, result.var, eps, decay, result.sorted_axis);

    const Array& gamma_reshaped = result.gamma;
    const Array& beta_reshaped = result.beta;

    Array out = fb->Forward(x.AsGradStopped(), gamma_reshaped.AsGradStopped(), beta_reshaped.AsGradStopped());
    internal::MakeViewForForwardBackwardOutput(out);

    BackwardBuilder bb{"batch_norm", {x, gamma_reshaped, beta_reshaped}, {out}};
    if (BackwardBuilder::Target bt = bb.CreateTarget({0, 1, 2})) {
        bt.Define([ fb = std::move(fb), x_tok = bb.RetainInput(0), gamma_tok = bb.RetainInput(1) ](BackwardContext & bctx) {
            const Array& gout = bctx.output_grad();
            std::array<Array, 3> ginputs = fb->Backward(gout.AsGradStopped());
            internal::MakeViewForForwardBackwardOutput(ginputs);
            const Array& gx = ginputs[0];
            const Array& ggamma = ginputs[1];
            const Array& gbeta = ginputs[2];
            assert(internal::GetArrayBody(gx)->nodes().empty());
            assert(internal::GetArrayBody(ggamma)->nodes().empty());
            assert(internal::GetArrayBody(gbeta)->nodes().empty());

            if (bctx.next_required()) {
                const Array& x = bctx.GetRetainedInput(x_tok);
                const Array& gamma_reshaped = bctx.GetRetainedInput(gamma_tok);
                BackwardBuilder bb2{"batch_norm_backward", {x, gamma_reshaped, gout}, {gx, ggamma, gbeta}};
                if (BackwardBuilder::Target bt2 = bb2.CreateTarget({0, 1, 2})) {
                    bt2.Define([fb, x, gamma](BackwardContext& bctx2) {
                        const Array& g2x = bctx2.output_grad(0);
                        const Array& g2gamma = bctx2.output_grad(1);
                        const Array& g2beta = bctx2.output_grad(2);

                        // const Array& x = *x_;
                        // const Array& gamma = *gamma_;
                        const Array& x_mean = *x_mean_;
                        const Array& x_inv_std = *x_inv_std_;

                        const Array& gout = *gout_;
                        const Array& gx = *gx_;
                        const Array& ggamma = *ggamma_;

                        // Auxiliary values
                        double inv_n = 1.0 / (x.GetTotalSize() / gamma.GetTotalSize());
                        Array r = (gx * ggx).Sum(axis_);
                        Array coeff = gamma * x_inv_std;
                        Array coeff_m = coeff * inv_n;
                        Array x_hat = (x - x_mean) * x_inv_std;

                        Array gggamma2 = gggamma - coeff_m * (x_hat * ggx).Sum(axis_);
                        Array ggbeta2 = ggbeta - coeff_m * ggx.Sum(axis_);

                        Array gx_hat2 = gggamma2 * gout - coeff_m * ggamma * ggx;
                        Array gstd2 = -x_inv_std * (r + (x_hat * gx_hat2).Sum(axis_));
                        Array gmean2 = -x_inv_std * gx_hat2.Sum(axis_);
                        Array gx2 = x_inv_std * gx_hat2 + inv_n * (gmean2 + x_hat * gstd2);
                        Array ggy2 = gggamma2 * x_hat + ggbeta2 + coeff * ggx;

                        Array ggamma2 = r / gamma;

                        return {std::move(gx2), std::move(ggamma2), std::move(ggy2)};
                        std::array<Array, 3> ginputs2 =
                                fb->DoubleBackward(g2x.AsGradStopped(), g2gamma.AsGradStopped(), g2beta.AsGradStopped());
                        internal::MakeViewForForwardBackwardOutput(ginputs2);
                        // TODO(niboshi): Make it further backproppable
                        // TODO(niboshi): Assign at once
                        bctx2.input_grad(0) = ginputs2[0];  // gx
                        bctx2.input_grad(1) = ginputs2[1];  // ggamma
                        bctx2.input_grad(2) = ginputs2[2];  // ggout
                    });
                }
                assert(bb2.is_complete());
            }

            // TODO(niboshi): Assign at once
            bctx.input_grad(0) = gx;
            bctx.input_grad(1) = ggamma;
            bctx.input_grad(2) = gbeta;
        });
    }
    assert(bb.is_complete());

    return out;
}

Array FixedBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const OptionalAxes& axis) {
    PreprocessBatchNormResult result =
            PreprocessBatchNorm(x, gamma.AsGradStopped(), beta.AsGradStopped(), mean.AsGradStopped(), var.AsGradStopped(), axis);
    {
        NoBackpropModeScope scope{};
        return x.device().FixedBatchNorm(x.AsGradStopped(), result.gamma, result.beta, result.mean, result.var, eps, result.sorted_axis);
    }
}

}  // namespace xchainer
