#include "chainerx/device.h"

#include <algorithm>
#include <type_traits>
#include <utility>

#include "chainerx/array.h"
#include "chainerx/context.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/statistics.h"
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
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const Axes& axis) {
    if (CHAINERX_DEBUG) {
        Shape reduced_shape = internal::ReduceShape(x.shape(), axis, true);
        CHAINERX_ASSERT(gamma.shape() == reduced_shape);
        CHAINERX_ASSERT(beta.shape() == reduced_shape);

        int64_t reduced_total_size = reduced_shape.GetTotalSize();
        CHAINERX_ASSERT(mean.GetTotalSize() == reduced_total_size);
        CHAINERX_ASSERT(var.GetTotalSize() == reduced_total_size);
    }
    Array inv_std = Reciprocal(Sqrt(var + eps));

    Array out = (x - mean) * inv_std * gamma + beta;

    return {std::move(out), std::move(inv_std)};
}

}  // namespace

GenericBatchNormForwardBackward::GenericBatchNormForwardBackward(
        const Array& running_mean, const Array& running_var, Scalar eps, Scalar decay, Axes axis)
    : running_mean_{running_mean}, running_var_{running_var}, eps_{eps}, decay_{decay}, axis_{std::move(axis)} {}

void GenericBatchNormForwardBackward::SetForwardResults(Array x, Array gamma, Array x_mean, Array x_inv_std) {
    CHAINERX_ASSERT(internal::GetArrayBody(x)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(gamma)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(x_mean)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(x_inv_std)->nodes().empty());
    x_ = std::make_shared<Array>(std::move(x));
    gamma_ = std::make_shared<Array>(std::move(gamma));
    x_mean_ = std::make_shared<Array>(std::move(x_mean));
    x_inv_std_ = std::make_shared<Array>(std::move(x_inv_std));
}

Array GenericBatchNormForwardBackward::Forward(const Array& x, const Array& gamma, const Array& beta) {
    CHAINERX_ASSERT(internal::GetArrayBody(x)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(gamma)->nodes().empty());
    CHAINERX_ASSERT(internal::GetArrayBody(beta)->nodes().empty());

    Array x_mean = Mean(x, axis_, true);
    Array x_var = Var(x, axis_, true);

    ApplyBatchNormResult result = ApplyBatchNorm(x, gamma, beta, x_mean, x_var, eps_, axis_);
    Array& out = result.out;
    Array& x_inv_std = result.inv_std;

    Scalar inv_decay = Scalar{1.0 - static_cast<double>(decay_)};
    int64_t n = x.GetTotalSize() / gamma.GetTotalSize();
    running_mean_ *= decay_;
    running_mean_ += inv_decay * x_mean;
    running_var_ *= decay_;
    running_var_ += inv_decay * (static_cast<double>(n) / std::max(n - 1, int64_t{1})) * x_var;

    SetForwardResults(x, gamma, std::move(x_mean), std::move(x_inv_std));

    return std::move(out);
}

std::array<Array, 3> GenericBatchNormForwardBackward::Backward(const Array& gout) {
    CHAINERX_ASSERT(internal::GetArrayBody(gout)->nodes().empty());

    // Note: x_inv_std_ has the information of eps.
    const Array& x = *x_;
    const Array& gamma = *gamma_;
    const Array& x_mean = *x_mean_;
    const Array& x_inv_std = *x_inv_std_;

    double inv_n = 1.0 / (x.GetTotalSize() / gamma.GetTotalSize());
    Array x_hat = (x - x_mean) * x_inv_std;
    Array ggamma = (gout * x_hat).Sum(axis_, true);
    Array gbeta = gout.Sum(axis_, true);
    Array gx = (gamma * x_inv_std) * (gout - (x_hat * ggamma + gbeta) * inv_n);

    return {std::move(gx), std::move(ggamma), std::move(gbeta)};
}

Array Device::FixedBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const Axes& axis) {
    ApplyBatchNormResult result = ApplyBatchNorm(x, gamma, beta, mean, var, eps, axis);
    return std::move(result.out);
}

}  // namespace chainerx
