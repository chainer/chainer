#include "xchainer/native/native_device.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/native/elementwise.h"
#include "xchainer/routines/creation.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace native {
namespace {

void Mean(const Array& a, const Axes& axis, const Array& out) {
    Device& device = a.device();
    device.Sum(a, axis, out);
    device.DivideAS(out, xchainer::internal::CountItemsAlongAxes(a.shape(), axis), out);
}

void Var(const Array& a, const Array& mean, const Axes& axis, const Array& out) {
    Array out_pre_reduction = EmptyLike(a, a.device());
    VisitDtype(out.dtype(), [&a, &mean, &out_pre_reduction](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T a, T mean, T& out) {
                T diff = a - mean;
                out = diff * diff;
            }
        };
        Elementwise<const T, const T, T>(Impl{}, a, mean.BroadcastTo(a.shape()), out_pre_reduction);
    });
    Mean(out_pre_reduction, axis, out);
}

class NativeBatchNormForwardBackward : public xchainer::BatchNormForwardBackward {
public:
    Array Forward(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& running_mean,
            const Array& running_var,
            Scalar eps,
            Scalar decay,
            const Axes& axis) override {
#ifndef NDEBUG
        {
            Shape reduced_shape = xchainer::internal::ReduceShape(x.shape(), axis, true);
            assert(gamma.shape() == reduced_shape);
            assert(beta.shape() == reduced_shape);

            int64_t reduced_total_size = reduced_shape.GetTotalSize();
            assert(running_mean.GetTotalSize() == reduced_total_size);
            assert(running_var.GetTotalSize() == reduced_total_size);
        }
#endif  // NDEBUG
        Dtype dtype = x.dtype();
        Device& device = x.device();

        Array x_mean = xchainer::internal::EmptyReduced(x.shape(), dtype, axis, true, device);
        Mean(x, axis, x_mean);

        Array x_var = xchainer::internal::EmptyReduced(x.shape(), dtype, axis, true, device);
        Var(x, x_mean, axis, x_var);

        Array out = EmptyLike(x, device);
        int64_t n = x.GetTotalSize() / gamma.GetTotalSize();
        VisitFloatingPointDtype(dtype, [&, eps, decay, n](auto pt) {
            using T = typename decltype(pt)::type;

            // Compute the batch normalization.
            struct BatchNormImpl {
                void operator()(int64_t /*i*/, T x, T x_mean, T x_var, T gamma, T beta, T& out) {
                    out = (x - x_mean) / std::sqrt(x_var + eps) * gamma + beta;
                }
                T eps;
            };
            Elementwise<const T, const T, const T, const T, const T, T>(
                    BatchNormImpl{static_cast<T>(eps)},
                    x,
                    x_mean.BroadcastTo(out.shape()),
                    x_var.BroadcastTo(out.shape()),
                    gamma.BroadcastTo(out.shape()),
                    beta.BroadcastTo(out.shape()),
                    out);

            // Update the running mean and variance in-place using an unbiased estimate.
            struct UpdateStatsImpl {
                void operator()(int64_t /*i*/, T mean, T var, T& running_mean, T& running_var) {
                    running_mean *= decay;
                    running_mean += (T{1} - decay) * mean;
                    running_var *= decay;
                    running_var += (T{1} - decay) * adjust * var;
                }
                T decay;
                T adjust;
            };
            Elementwise<const T, const T, T, T>(
                    UpdateStatsImpl{static_cast<T>(decay), static_cast<T>(n) / std::max(n - 1, int64_t{1})},
                    x_mean.Reshape(running_mean.shape()),
                    x_var.Reshape(running_mean.shape()),
                    running_mean,
                    running_var);
        });
        return out;
    }

    // TODO(hvy): Implement me.
    std::array<Array, 3> Backward(
            const Array& /*x*/, const Array& /*gamma*/, const Array& /*gout*/, Scalar /*eps*/, Scalar /*decay*/, const Axes& /*axis*/)
            override {
        return {Array{}, Array{}, Array{}};
    }
};

}  // namespace

std::unique_ptr<BatchNormForwardBackward> NativeDevice::GetBatchNormForwardBackward() {
    return std::make_unique<NativeBatchNormForwardBackward>();
}

}  // namespace native
}  // namespace xchainer
