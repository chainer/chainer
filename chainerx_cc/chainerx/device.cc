#include "chainerx/device.h"

#include <algorithm>
#include <type_traits>
#include <utility>

#include "chainerx/array.h"
#include "chainerx/context.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/statistics.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/thread_local_state.h"

namespace chainerx {

void Device::CheckDevicesCompatible(const Array& array) {
    if (this != &array.device()) {
        throw DeviceError{"Device (", name(), ") is not compatible with array's device (", array.device().name(), ")."};
    }
}

namespace internal {

Device* GetDefaultDeviceNoExcept() noexcept { return internal::GetInternalThreadLocalState().default_device; }

}  // namespace internal

Device& GetDefaultDevice() {
    Device*& default_device = internal::GetInternalThreadLocalState().default_device;
    if (default_device == nullptr) {
        default_device = &GetDefaultContext().GetDevice({native::NativeBackend::kDefaultName, 0});
    }
    return *default_device;
}

void SetDefaultDevice(Device* device) {
    if (device != nullptr && &device->backend().context() != &GetDefaultContext()) {
        throw ContextError{"Context mismatch between default device and default context."};
    }
    Device*& default_device = internal::GetInternalThreadLocalState().default_device;
    default_device = device;
}

void CheckEqual(const Device& lhs, const Device& rhs) {
    if (&lhs != &rhs) {
        throw DeviceError{"Devices do not match: ", lhs.name(), ", ", rhs.name(), "."};
    }
}

namespace {

struct ApplyBatchNormResult {
    Array out;
    Array inv_std;
};

ApplyBatchNormResult ApplyBatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& mean,
        const Array& var,
        Scalar eps,
        const Axes& axis,
        Dtype interm_dtype) {
    if (CHAINERX_DEBUG) {
        Shape reduced_shape = internal::ReduceShape(x.shape(), axis, true);
        CHAINERX_ASSERT(gamma.shape() == reduced_shape);
        CHAINERX_ASSERT(beta.shape() == reduced_shape);

        int64_t reduced_total_size = reduced_shape.GetTotalSize();
        CHAINERX_ASSERT(mean.GetTotalSize() == reduced_total_size);
        CHAINERX_ASSERT(var.GetTotalSize() == reduced_total_size);
    }
    // TODO(hvy): Avoid AsType by passing dtype arguments to the following routines to minimize copies.
    const Array& x_cast = x.AsType(interm_dtype, false);
    const Array& gamma_cast = gamma.AsType(interm_dtype, false);
    const Array& beta_cast = beta.AsType(interm_dtype, false);
    const Array& mean_cast = mean.AsType(interm_dtype, false);
    const Array& var_cast = var.AsType(interm_dtype, false);

    Array inv_std = Reciprocal(Sqrt(var_cast + eps));

    Array out = (x_cast - mean_cast) * inv_std * gamma_cast + beta_cast;

    return {out.dtype() == x.dtype() ? std::move(out) : out.AsType(x.dtype()), std::move(inv_std)};
}

}  // namespace

GenericBatchNormForwardBackward::GenericBatchNormForwardBackward(
        const Array& running_mean, const Array& running_var, Scalar eps, Scalar decay, Axes axis)
    : running_mean_{running_mean}, running_var_{running_var}, eps_{eps}, decay_{decay}, axis_{std::move(axis)} {}

void GenericBatchNormForwardBackward::SetForwardResults(Array x, Array gamma, Array x_mean, Array x_inv_std, Dtype beta_dtype) {
    CHAINERX_ASSERT(internal::GetArrayBody(x)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(gamma)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(x_mean)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(x_inv_std)->nodes().empty());
    x_ = std::make_shared<Array>(std::move(x));
    gamma_ = std::make_shared<Array>(std::move(gamma));
    x_mean_ = std::make_shared<Array>(std::move(x_mean));
    x_inv_std_ = std::make_shared<Array>(std::move(x_inv_std));
    beta_dtype_ = beta_dtype;
}

Array GenericBatchNormForwardBackward::Forward(const Array& x, const Array& gamma, const Array& beta) {
    CHAINERX_ASSERT(internal::GetArrayBody(x)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(gamma)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(beta)->nodes().empty());

    CHAINERX_ASSERT(GetKind(x.dtype()) == DtypeKind::kFloat);
    CHAINERX_ASSERT(GetKind(gamma.dtype()) == DtypeKind::kFloat);
    CHAINERX_ASSERT(GetKind(beta.dtype()) == DtypeKind::kFloat);
    CHAINERX_ASSERT(GetKind(running_mean().dtype()) == DtypeKind::kFloat);
    CHAINERX_ASSERT(GetKind(running_var().dtype()) == DtypeKind::kFloat);

    // Compute the mean and variance of x with promoted dtype if the parameters have higher precisions.
    Dtype interm_dtype = ResultType(x, gamma, beta);
    const Array& x_cast = x.dtype() == interm_dtype ? x : x.AsType(interm_dtype);
    Array x_mean = Mean(x_cast, axis_, true);
    Array x_var = Var(x_cast, axis_, true);

    ApplyBatchNormResult result = ApplyBatchNorm(x, gamma, beta, x_mean, x_var, eps_, axis_, interm_dtype);
    Array& x_inv_std = result.inv_std;

    Scalar inv_decay = Scalar{1.0 - static_cast<double>(decay_)};
    int64_t n = x.GetTotalSize() / gamma.GetTotalSize();

    // TODO(hvy): Avoid AsType when IAdd supports mixed dtypes.
    running_mean_ *= decay_;
    running_mean_ += (inv_decay * x_mean).AsType(running_mean_.dtype(), false);
    running_var_ *= decay_;
    running_var_ += (inv_decay * (static_cast<double>(n) / std::max(n - 1, int64_t{1})) * x_var).AsType(running_var_.dtype(), false);

    SetForwardResults(x, gamma, std::move(x_mean), std::move(x_inv_std), beta.dtype());

    return std::move(result.out);
}

std::array<Array, 3> GenericBatchNormForwardBackward::Backward(const Array& gout) {
    CHAINERX_ASSERT(internal::GetArrayBody(gout)->nodes().empty());

    // Note: x_inv_std_ has the information of eps.
    const Array& x = *x_;
    const Array& gamma = *gamma_;
    const Array& x_mean = *x_mean_;  // Promoted dtype.
    const Array& x_inv_std = *x_inv_std_;  // Promoted dtype.

    Dtype interm_dtype = x_mean.dtype();

    int64_t n = x.GetTotalSize() / gamma.GetTotalSize();
    double inv_n = 1.0 / n;
    // TODO(hvy): Avoid AsType.
    Array gout_cast = gout.AsType(interm_dtype, false);
    Array x_hat = (x.AsType(interm_dtype, false) - x_mean) * x_inv_std;
    Array ggamma = (gout_cast * x_hat).Sum(axis_, true);
    Array gbeta = gout_cast.Sum(axis_, true);
    Array gx = (gamma.AsType(interm_dtype, false) * x_inv_std) * (gout_cast - (x_hat * ggamma + gbeta) * inv_n);

    if (gx.dtype() != x.dtype()) {
        gx = gx.AsType(x.dtype());
    }
    if (ggamma.dtype() != gamma.dtype()) {
        ggamma = ggamma.AsType(gamma.dtype());
    }
    if (gbeta.dtype() != beta_dtype()) {
        gbeta = gbeta.AsType(beta_dtype());
    }

    return {std::move(gx), std::move(ggamma), std::move(gbeta)};
}

Array Device::FixedBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const Axes& axis) {
    Dtype interm_dtype = ResultType(x, gamma, beta, mean, var);
    ApplyBatchNormResult result = ApplyBatchNorm(x, gamma, beta, mean, var, eps, axis, interm_dtype);
    return std::move(result.out);
}

}  // namespace chainerx
