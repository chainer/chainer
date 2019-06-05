#include "chainerx/routines/normalization.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/normalization.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/statistics.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace {

struct PreprocessBatchNormResult {
    // Arrays are reshaped if necessary
    Array gamma;
    Array beta;
    Array mean;
    Array var;
    Axes sorted_axis;
};

// Reshapes the array. If the shape is unchanged, an array with identical array body is returned. Note that chainerx::Reshape() returns
// a view with different array body if the shape is unchanged.
Array ReshapeOrIdentity(const Array& a, const Shape& shape) {
    if (a.shape() == shape) {
        return a;
    }
    return a.Reshape(shape);
}

void CheckBatchNormSupportedKind(const Array& array) {
    // BatchNorm only supports inputs of float kind.
    if (GetKind(array.dtype()) != DtypeKind::kFloat) {
        throw DtypeError{"BatchNorm only supports floating kind inputs."};
    }
}

// Reshapes the input arrays (except x) as needed.
// Sorted axes is also returned.
PreprocessBatchNormResult PreprocessBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, const OptionalAxes& axis) {
    CheckBatchNormSupportedKind(x);
    CheckBatchNormSupportedKind(gamma);
    CheckBatchNormSupportedKind(beta);
    CheckBatchNormSupportedKind(mean);
    CheckBatchNormSupportedKind(var);

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
    CHAINERX_ASSERT(gamma_reshaped.data() == gamma.data());  // No data copy should occur
    CHAINERX_ASSERT(beta_reshaped.data() == beta.data());
    CHAINERX_ASSERT(mean_reshaped.data() == mean.data());
    CHAINERX_ASSERT(var_reshaped.data() == var.data());

    return {std::move(gamma_reshaped), std::move(beta_reshaped), std::move(mean_reshaped), std::move(var_reshaped), sorted_axis};
}

Array ArrayOrZeros(const nonstd::optional<Array>& array, const Array& zeros_template, Dtype dtype) {
    if (array.has_value()) {
        if (array->dtype() == dtype) {
            return *array;
        }
        return array->AsType(dtype);
    }
    return Zeros(zeros_template.shape(), dtype, zeros_template.device());
}

std::tuple<Array, std::unique_ptr<BatchNormGradState>> ApplyGenericBatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& mean,
        const Array& var,
        Scalar eps,
        const Axes& axis,
        Dtype interm_dtype,
        bool return_state,
        const nonstd::optional<Array>& out) {
    if (CHAINERX_DEBUG) {
        Shape reduced_shape = internal::ReduceShape(x.shape(), axis, true);
        CHAINERX_ASSERT(gamma.shape() == reduced_shape);
        CHAINERX_ASSERT(beta.shape() == reduced_shape);

        int64_t reduced_total_size = reduced_shape.GetTotalSize();
        CHAINERX_ASSERT(mean.GetTotalSize() == reduced_total_size);
        CHAINERX_ASSERT(var.GetTotalSize() == reduced_total_size);
    }
    // TODO(hvy): Implement and test the `out` argument.
    if (out.has_value()) {
        throw NotImplementedError{"Passing out as an argument is not yet supported."};
    }

    // TODO(hvy): Avoid `AsType` by passing dtype arguments to the following routines to minimize copies.
    const Array& x_cast = x.AsType(interm_dtype, false);
    const Array& gamma_cast = gamma.AsType(interm_dtype, false);
    const Array& beta_cast = beta.AsType(interm_dtype, false);
    Array mean_cast = mean.AsType(interm_dtype, false);
    const Array& var_cast = var.AsType(interm_dtype, false);

    Array inv_std = Reciprocal(Sqrt(var_cast + eps));
    Array out_cast = (x_cast - mean_cast) * inv_std * gamma_cast + beta_cast;
    const Array& actual_out = out_cast.dtype() == x.dtype() ? out_cast : out_cast.AsType(x.dtype());

    std::unique_ptr<BatchNormGradState> state =
            return_state ? std::make_unique<GenericBatchNormGradState>(std::move(mean_cast), std::move(inv_std), beta.dtype()) : nullptr;

    return std::make_tuple(actual_out, std::move(state));
}

}  // namespace

std::tuple<Array, std::unique_ptr<BatchNormGradState>> GenericBatchNormKernel::Call(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        Scalar eps,
        Scalar decay,
        const Axes& axis,
        bool return_state,
        const nonstd::optional<Array>& out) {
    CHAINERX_ASSERT(internal::GetArrayBody(x)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(gamma)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(beta)->nodes().empty());
    CHAINERX_ASSERT(GetKind(x.dtype()) == DtypeKind::kFloat);
    CHAINERX_ASSERT(GetKind(gamma.dtype()) == DtypeKind::kFloat);
    CHAINERX_ASSERT(GetKind(beta.dtype()) == DtypeKind::kFloat);
    CHAINERX_ASSERT(GetKind(running_mean.dtype()) == DtypeKind::kFloat);
    CHAINERX_ASSERT(GetKind(running_var.dtype()) == DtypeKind::kFloat);

    // Compute the mean and variance of x with promoted dtype if the parameters have higher precisions.
    Dtype interm_dtype = ResultType(x, gamma, beta);
    const Array& x_cast = x.dtype() == interm_dtype ? x : x.AsType(interm_dtype);
    Array x_mean = Mean(x_cast, axis, true);
    Array x_var = Var(x_cast, axis, true);
    std::tuple<Array, std::unique_ptr<BatchNormGradState>> result =
            ApplyGenericBatchNorm(x, gamma, beta, x_mean, x_var, eps, axis, interm_dtype, return_state, out);

    // Update running values.
    // TODO(hvy): Avoid `AsType` when `IAdd` supports mixed dtypes.
    Scalar inv_decay = Scalar{1.0 - static_cast<double>(decay)};
    int64_t n = x.GetTotalSize() / gamma.GetTotalSize();
    running_mean *= decay;
    running_mean += (inv_decay * x_mean).AsType(running_mean.dtype(), false);
    running_var *= decay;
    running_var += (inv_decay * (static_cast<double>(n) / std::max(n - 1, int64_t{1})) * x_var).AsType(running_var.dtype(), false);

    return result;
}

std::tuple<Array, Array, Array> GenericBatchNormGradKernel::Call(
        const Array& x,
        const Array& gamma,
        const Array& gout,
        Scalar /*eps*/,
        const Axes& axis,
        const std::shared_ptr<BatchNormGradState>& state,
        const nonstd::optional<Array>& gx,
        const nonstd::optional<Array>& ggamma,
        const nonstd::optional<Array>& gbeta) {
    // TODO(hvy): Implement and test the `gx` argument.
    if (gx.has_value()) {
        throw NotImplementedError{"Passing gx as an argument is not yet supported."};
    }
    // TODO(hvy): Implement and test the `ggamma` argument.
    if (ggamma.has_value()) {
        throw NotImplementedError{"Passing ggamma as an argument is not yet supported."};
    }
    // TODO(hvy): Implement and test the `gbeta` argument.
    if (gbeta.has_value()) {
        throw NotImplementedError{"Passing gbeta as an argument is not yet supported."};
    }

    // TODO(hvy): Implement recomputation of x_mean and x_inv_std in case they are not given by the state.
    CHAINERX_ASSERT(state != nullptr);
    auto generic_state = dynamic_cast<GenericBatchNormGradState&>(*state);
    // x_mean and x_inv_std have promoted dtypes.
    const Array& x_mean = generic_state.x_mean();
    const Array& x_inv_std = generic_state.x_inv_std();  // Note: x_inv_std_ has the information of eps.
    Dtype beta_dtype = generic_state.beta_dtype();

    // TODO(hvy): Avoid `AsType`.
    Dtype interm_dtype = x_mean.dtype();
    int64_t n = x.GetTotalSize() / gamma.GetTotalSize();
    double inv_n = 1.0 / n;
    Array gout_cast = gout.AsType(interm_dtype, false);
    Array x_hat = (x.AsType(interm_dtype, false) - x_mean) * x_inv_std;
    Array actual_ggamma = (gout_cast * x_hat).Sum(axis, true);
    Array actual_gbeta = gout_cast.Sum(axis, true);
    Array actual_gx = (gamma.AsType(interm_dtype, false) * x_inv_std) * (gout_cast - (x_hat * actual_ggamma + actual_gbeta) * inv_n);

    if (actual_gx.dtype() != x.dtype()) {
        actual_gx = actual_gx.AsType(x.dtype());
    }
    if (actual_ggamma.dtype() != gamma.dtype()) {
        actual_ggamma = actual_ggamma.AsType(gamma.dtype());
    }
    if (actual_gbeta.dtype() != beta_dtype) {
        actual_gbeta = actual_gbeta.AsType(beta_dtype);
    }

    return std::make_tuple(std::move(actual_gx), std::move(actual_ggamma), std::move(actual_gbeta));
}

Array GenericFixedBatchNormKernel::Call(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& mean,
        const Array& var,
        Scalar eps,
        const Axes& axis,
        const nonstd::optional<Array>& out) {
    Dtype interm_dtype = ResultType(x, gamma, beta, mean, var);
    std::tuple<Array, std::unique_ptr<BatchNormGradState>> result =
            ApplyGenericBatchNorm(x, gamma, beta, mean, var, eps, axis, interm_dtype, false, out);
    return out.has_value() ? *out : std::get<0>(result);
}

Array BatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        Scalar eps,
        Scalar decay,
        const OptionalAxes& axis) {
    // Preprocess inputs.
    PreprocessBatchNormResult result = PreprocessBatchNorm(x, gamma, beta, running_mean, running_var, axis);
    const Array& gamma_reshaped = result.gamma;
    const Array& beta_reshaped = result.beta;
    const Array& mean_reshaped = result.mean;
    const Array& var_reshaped = result.var;
    const Axes& sorted_axis = result.sorted_axis;

    Device& device = x.device();

    // Compute forward.
    Array out{};
    std::shared_ptr<BatchNormGradState> state{};
    {
        NoBackpropModeScope scope{};
        std::tie(out, state) = device.backend().CallKernel<BatchNormKernel>(
                x.AsGradStopped(),
                gamma_reshaped.AsGradStopped(),
                beta_reshaped.AsGradStopped(),
                mean_reshaped,
                var_reshaped,
                eps,
                decay,
                sorted_axis,
                true,
                nonstd::nullopt);
    }
    CHAINERX_ASSERT(state != nullptr);

    internal::MakeViewForForwardBackwardOutput(out);

    BackwardBuilder bb{"batch_norm", {x, gamma_reshaped, beta_reshaped}, {out}};
    if (BackwardBuilder::Target bt = bb.CreateTarget({0, 1, 2})) {
        bt.Define([state = std::move(state),
                   x_tok = bb.RetainInput(0),
                   gamma_tok = bb.RetainInput(1),
                   eps,
                   sorted_axis,
                   beta_shape = beta_reshaped.shape(),
                   beta_dtype = beta_reshaped.dtype()](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& x = bctx.GetRetainedInput(x_tok);
            const Array& gamma_reshaped = bctx.GetRetainedInput(gamma_tok);

            Device& device = x.device();

            // Compute backward.
            Array gx{};
            Array ggamma{};
            Array gbeta{};
            {
                NoBackpropModeScope scope{};
                std::tie(gx, ggamma, gbeta) = device.backend().CallKernel<BatchNormGradKernel>(
                        x, gamma_reshaped, gout, eps, sorted_axis, state, nonstd::nullopt, nonstd::nullopt, nonstd::nullopt);
            }
            CHAINERX_ASSERT(internal::GetArrayBody(gx)->nodes().empty());
            CHAINERX_ASSERT(internal::GetArrayBody(ggamma)->nodes().empty());
            CHAINERX_ASSERT(internal::GetArrayBody(gbeta)->nodes().empty());

            if (bctx.next_required()) {
                BackwardBuilder bb2{"batch_norm_backward", {x, gamma_reshaped, gout}, {gx, ggamma, gbeta}};
                if (BackwardBuilder::Target bt2 = bb2.CreateTarget({0, 1, 2})) {
                    bt2.Define([x_tok = bb2.RetainInput(0),
                                gamma2_tok = bb2.RetainInput(1),
                                gout_tok = bb2.RetainInput(2),
                                eps,
                                sorted_axis,
                                gx_tok = bb2.RetainOutput(0),
                                ggamma_tok = bb2.RetainOutput(1)](BackwardContext& bctx2) {
                        const Array& x_retained = bctx2.GetRetainedInput(x_tok);
                        const Array& gamma_reshaped_retained = bctx2.GetRetainedInput(gamma2_tok);
                        const Array& gout_retained = bctx2.GetRetainedInput(gout_tok);

                        // TODO(hvy): Avoid AsType by passing dtype arguments to Mean, Var, etc. to minimize copies.
                        Dtype interm_dtype = ResultType(gout_retained, x_retained, gamma_reshaped_retained);
                        const Array& x = x_retained.AsType(interm_dtype, false);
                        const Array& gamma_reshaped = gamma_reshaped_retained.AsType(interm_dtype, false);
                        const Array& gout = gout_retained.AsType(interm_dtype, false);

                        Array ggx = ArrayOrZeros(bctx2.output_grad(0), x, interm_dtype);
                        Array gggamma = ArrayOrZeros(bctx2.output_grad(1), gamma_reshaped, interm_dtype);
                        Array ggbeta = ArrayOrZeros(bctx2.output_grad(2), gamma_reshaped, interm_dtype);

                        const Array& x_mean = Mean(x, sorted_axis, true).AsType(interm_dtype, false);
                        const Array& x_var = Var(x, sorted_axis, true).AsType(interm_dtype, false);
                        const Array& x_inv_std = Reciprocal(Sqrt(x_var + eps)).AsType(interm_dtype, false);

                        const Array& gx = bctx2.GetRetainedOutput(gx_tok).AsType(interm_dtype, false);
                        const Array& ggamma = bctx2.GetRetainedOutput(ggamma_tok).AsType(interm_dtype, false);

                        // Auxiliary values
                        int64_t n = x.GetTotalSize() / gamma_reshaped.GetTotalSize();
                        double inv_n = 1.0 / n;
                        Array r = (gx * ggx).Sum(sorted_axis, true);
                        Array coeff = gamma_reshaped * x_inv_std;
                        Array coeff_m = coeff * inv_n;
                        Array x_hat = (x - x_mean) * x_inv_std;

                        Array gggamma2 = gggamma - coeff_m * (x_hat * ggx).Sum(sorted_axis, true);
                        Array ggbeta2 = ggbeta - coeff_m * ggx.Sum(sorted_axis, true);

                        Array gx_hat2 = gggamma2 * gout - coeff_m * ggamma * ggx;
                        Array gstd2 = -x_inv_std * (r + (x_hat * gx_hat2).Sum(sorted_axis, true));
                        Array gmean2 = -x_inv_std * gx_hat2.Sum(sorted_axis, true);
                        Array gx2 = x_inv_std * gx_hat2 + inv_n * (gmean2 + x_hat * gstd2);
                        Array ggout2 = gggamma2 * x_hat + ggbeta2 + coeff * ggx;

                        Array ggamma2 = r / gamma_reshaped;

                        if (gx2.dtype() != x_retained.dtype()) {
                            gx2 = gx2.AsType(x_retained.dtype());
                        }
                        if (ggamma2.dtype() != gamma_reshaped_retained.dtype()) {
                            ggamma2 = ggamma2.AsType(gamma_reshaped_retained.dtype());
                        }

                        if (ggout2.dtype() != gout_retained.dtype()) {
                            ggout2 = ggout2.AsType(gout_retained.dtype());
                        }

                        bctx2.input_grad(0) = std::move(gx2);
                        bctx2.input_grad(1) = std::move(ggamma2);
                        bctx2.input_grad(2) = std::move(ggout2);
                    });
                }
                bb2.Finalize();
            }

            // TODO(niboshi): Assign at once
            bctx.input_grad(0) = std::move(gx);
            bctx.input_grad(1) = std::move(ggamma);
            bctx.input_grad(2) = std::move(gbeta);
        });
    }
    bb.Finalize();

    return out;
}

Array FixedBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const OptionalAxes& axis) {
    PreprocessBatchNormResult result =
            PreprocessBatchNorm(x, gamma.AsGradStopped(), beta.AsGradStopped(), mean.AsGradStopped(), var.AsGradStopped(), axis);

    {
        NoBackpropModeScope scope{};
        return x.device().backend().CallKernel<FixedBatchNormKernel>(
                x.AsGradStopped(), result.gamma, result.beta, result.mean, result.var, eps, result.sorted_axis, nonstd::nullopt);
    }
}

}  // namespace chainerx
