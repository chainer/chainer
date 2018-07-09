#include "xchainer/device.h"

#include <algorithm>
#include <type_traits>
#include <utility>

#include "xchainer/array.h"
#include "xchainer/context.h"
#include "xchainer/error.h"
#include "xchainer/native/native_backend.h"
#include "xchainer/routines/creation.h"

namespace xchainer {
namespace {

thread_local Device* t_default_device = nullptr;
static_assert(std::is_pod<decltype(t_default_device)>::value, "t_default_device must be POD");

}  // namespace

void Device::CheckDevicesCompatible(const Array& array) {
    if (this != &array.device()) {
        throw DeviceError{"Device (", name(), ") is not compatible with array's device (", array.device().name(), ")."};
    }
}

namespace internal {

Device* GetDefaultDeviceNoExcept() noexcept { return t_default_device; }

}  // namespace internal

Device& GetDefaultDevice() {
    if (t_default_device == nullptr) {
        t_default_device = &GetDefaultContext().GetDevice({native::NativeBackend::kDefaultName, 0});
    }
    return *t_default_device;
}

void SetDefaultDevice(Device* device) {
    if (device != nullptr && &device->backend().context() != &GetDefaultContext()) {
        throw ContextError{"Context mismatch between default device and default context."};
    }
    t_default_device = device;
}

void CheckEqual(const Device& lhs, const Device& rhs) {
    if (&lhs != &rhs) {
        throw DeviceError{"Devices do not match: ", lhs.name(), ", ", rhs.name(), "."};
    }
}

namespace {

// Differentiable mean.
// TODO(niboshi): Move to routines
Array Mean(const Array& a, const Axes& axis, bool keepdims) {
    return a.Sum(axis, keepdims) / xchainer::internal::CountItemsAlongAxes(a.shape(), axis);
}

// Differentiable variance.
// TODO(niboshi): Move to routines
Array Var(const Array& a, const Array& mean, const Axes& axis, bool keepdims) {
    Array diff = a - mean;
    return Mean(diff * diff, axis, keepdims);
}

// Differentiable sqrt.
// TODO(niboshi): Move to routines
Array Sqrt(const Array& x) {
    Array out = EmptyLike(x, x.device());
    x.device().Sqrt(x, out);
    // TODO(niboshi): Implement backward
    return out;
}

// Differentiable reciprocal.
// TODO(niboshi): Move to routines
Array Reciprocal(const Array& x) { return OnesLike(x, x.device()) / x; }

struct ApplyBatchNormResult {
    Array out;
    Array inv_std;
};

ApplyBatchNormResult ApplyBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const Axes& axis) {
#ifndef NDEBUG
    {
        Shape reduced_shape = xchainer::internal::ReduceShape(x.shape(), axis, true);
        assert(gamma.shape() == reduced_shape);
        assert(beta.shape() == reduced_shape);

        int64_t reduced_total_size = reduced_shape.GetTotalSize();
        assert(mean.GetTotalSize() == reduced_total_size);
        assert(var.GetTotalSize() == reduced_total_size);
    }
#else
    (void)axis;  // unused
#endif  // NDEBUG
    Array inv_std = Reciprocal(Sqrt(var + eps));

    Array out = (x - mean) * inv_std * gamma + beta;

    assert(!out.IsGradRequired(AnyGraph{}));
    assert(!inv_std.IsGradRequired(AnyGraph{}));
    return {std::move(out), std::move(inv_std)};
}

}  // namespace

GenericBatchNormForwardBackward::GenericBatchNormForwardBackward(
        const Array& running_mean, const Array& running_var, Scalar eps, Scalar decay, Axes axis)
    : running_mean_{running_mean}, running_var_{running_var}, eps_{eps}, decay_{decay}, axis_{std::move(axis)} {}

void GenericBatchNormForwardBackward::SetForwardResults(Array x, Array gamma, Array x_mean, Array x_inv_std) {
    x_ = std::make_shared<Array>(std::move(x));
    gamma_ = std::make_shared<Array>(std::move(gamma));
    x_mean_ = std::make_shared<Array>(std::move(x_mean));
    x_inv_std_ = std::make_shared<Array>(std::move(x_inv_std));
}

void GenericBatchNormForwardBackward::SetBackwardResults(Array gout, Array gx, Array ggamma) {
    gout_ = std::make_shared<Array>(std::move(gout));
    gx_ = std::make_shared<Array>(std::move(gx));
    ggamma_ = std::make_shared<Array>(std::move(ggamma));
}

Array GenericBatchNormForwardBackward::Forward(const Array& x, const Array& gamma, const Array& beta) {
    Array x_mean = Mean(x, axis_, true);
    Array x_var = Var(x, x_mean, axis_, true);

    ApplyBatchNormResult result = ApplyBatchNorm(x, gamma, beta, x_mean, x_var, eps_, axis_);
    Array& out = result.out;
    Array& x_inv_std = result.inv_std;

    Scalar inv_decay = Scalar{1.0 - static_cast<double>(decay_)};
    int64_t n = x.GetTotalSize() / gamma.GetTotalSize();
    running_mean_ *= decay_;
    running_mean_ += inv_decay * x_mean;
    running_var_ *= decay_;
    running_var_ += inv_decay * (static_cast<double>(n) / std::max(n - 1, int64_t{1})) * x_var;

    SetForwardResults(x.AsGradStopped(), gamma.AsGradStopped(), std::move(x_mean), std::move(x_inv_std));

    return std::move(out);
}

std::array<Array, 3> GenericBatchNormForwardBackward::Backward(const Array& gout) {
    // Note: x_inv_std_ has the information of eps.
    const Array& x = *x_;
    const Array& gamma = *gamma_;
    const Array& x_mean = *x_mean_;
    const Array& x_inv_std = *x_inv_std_;

    double inv_n = 1.0 / (x.GetTotalSize() / gamma.GetTotalSize());
    Array x_hat = (x - x_mean) * x_inv_std;
    Array ggamma = (gout * x_hat).Sum(axis_);
    Array gbeta = gout.Sum(axis_);
    Array gx = (gamma * x_inv_std) * (gout - (x_hat * ggamma + gbeta) * inv_n);

    SetBackwardResults(gout.AsGradStopped(), gx, ggamma);

    return {std::move(gx), std::move(ggamma), std::move(gbeta)};
}

std::array<Array, 3> GenericBatchNormForwardBackward::DoubleBackward(const Array& ggx, const Array& gggamma, const Array& ggbeta) {
    const Array& x = *x_;
    const Array& gamma = *gamma_;
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
}

Array Device::FixedBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const Axes& axis) {
    ApplyBatchNormResult result = ApplyBatchNorm(x, gamma, beta, mean, var, eps, axis);
    assert(!result.out.IsGradRequired(AnyGraph{}));
    return std::move(result.out);
}

}  // namespace xchainer
