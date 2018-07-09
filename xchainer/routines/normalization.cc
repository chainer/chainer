#include "xchainer/routines/normalization.h"

#include <cstdint>
#include <memory>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/backward.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
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

    Array out = fb->Forward(x, result.gamma, result.beta);

    if (x.IsGradRequired(AnyGraph{}) || gamma.IsGradRequired(AnyGraph{}) || beta.IsGradRequired(AnyGraph{})) {
        BackwardBuilder bb{"batch_norm", {out}};
        bb.Define({x, gamma, beta}, [ fb = std::move(fb), x, gamma = result.gamma ](BackwardContext & bctx) {
            const Array& gout = bctx.output_grad();
            std::array<Array, 3> ginputs = fb->Backward(gout);
            const Array& gx = ginputs[0];
            const Array& ggamma = ginputs[1];
            const Array& gbeta = ginputs[2];
            assert(gx.IsConstant());
            assert(ggamma.IsConstant());
            assert(gbeta.IsConstant());

            Array x_cut = bctx.Cut(x);
            Array gamma_cut = bctx.Cut(gamma);

            if (bctx.next_required() &&
                (x_cut.IsGradRequired(AnyGraph{}) || gamma_cut.IsGradRequired(AnyGraph{}) || gout.IsGradRequired(AnyGraph{}))) {
                BackwardBuilder bb2{"batch_norm_backward", {gx, ggamma, gbeta}};
                bb2.Define({x_cut, gamma_cut, gout}, [fb](BackwardContext& bctx2) {
                    const Array& g2x = bctx2.output_grad(0);
                    const Array& g2gamma = bctx2.output_grad(1);
                    const Array& g2beta = bctx2.output_grad(2);
                    std::array<Array, 3> ginputs2 = fb->DoubleBackward(g2x, g2gamma, g2beta);
                    // TODO(niboshi): Make it further backproppable
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

Array FixedBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const OptionalAxes& axis) {
    PreprocessBatchNormResult result =
            PreprocessBatchNorm(x, gamma.AsGradStopped(), beta.AsGradStopped(), mean.AsGradStopped(), var.AsGradStopped(), axis);
    return x.device().FixedBatchNorm(x.AsGradStopped(), result.gamma, result.beta, result.mean, result.var, eps, result.sorted_axis);
}

}  // namespace xchainer
