#include "xchainer/device.h"

#include <algorithm>
#include <tuple>
#include <type_traits>

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
    Array x_mean;
    Array x_var;
    Array x_inv_std;
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
#endif  // NDEBUG
    Array x_const = x.AsConstant();
    Array x_mean = Mean(x_const, axis, true);
    Array x_var = Var(x_const, x_mean, axis, true);
    Array x_inv_std = Reciprocal(Sqrt(x_var + eps));

    Array out = (x_const - x_mean) * x_inv_std * gamma.AsConstant() + beta.AsConstant();

    return {std::move(out), std::move(x_mean), std::move(x_var), std::move(x_inv_std)};
}

}  // namespace

Array GenericBatchNormForwardBackward::Forward(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        Scalar eps,
        Scalar decay,
        const Axes& axis) {
    ApplyBatchNormResult result = ApplyBatchNorm(x, gamma, beta, running_mean, running_var, eps, axis);
    Array& out = result.out;
    Array& x_mean = result.x_mean;
    Array& x_var = result.x_var;
    Array& x_inv_std = result.x_inv_std;

    Scalar inv_decay = Scalar{1.0 - static_cast<double>(decay)};
    int64_t n = x.GetTotalSize() / gamma.GetTotalSize();
    running_mean *= decay;
    running_mean += inv_decay * x_mean;
    running_var *= decay;
    running_var += inv_decay * (static_cast<double>(n) / std::max(n - 1, int64_t{1})) * x_var;

    x_mean_ = std::make_shared<Array>(x_mean);
    x_inv_std_ = std::make_shared<Array>(x_inv_std);
    return std::move(out);
}

std::array<Array, 3> GenericBatchNormForwardBackward::Backward(
        const Array& x, const Array& gamma, const Array& gout, Scalar /*eps*/, const Axes& axis) {
    // Note: x_inv_std_ has the information of eps.
    const Array& x_mean = *x_mean_;
    const Array& x_inv_std = *x_inv_std_;
    double inv_n = 1.0 / (x.GetTotalSize() / gamma.GetTotalSize());
    Array x_hat = (x - x_mean) * x_inv_std;
    Array ggamma = (gout * x_hat).Sum(axis);
    Array gbeta = gout.Sum(axis);
    Array gx = (gamma * x_inv_std) * (gout - (x_hat * ggamma + gbeta) * inv_n);
    return {{gx, ggamma, gbeta}};
}

Array Device::FixedBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const Axes& axis) {
    ApplyBatchNormResult result = ApplyBatchNorm(x, gamma, beta, mean, var, eps, axis);
    return std::move(result.out);
}

}  // namespace xchainer
