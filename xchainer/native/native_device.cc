#include "xchainer/native/native_device.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gsl/gsl>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/axes.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/elementwise.h"
#include "xchainer/native/reduce.h"
#include "xchainer/numeric_limits.h"
#include "xchainer/reduction_kernel_arg.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/routines/math.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/slice.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace native {

namespace {

Array ExpandDims(const Array& a, const Axes& axes) {
    // Compute the new set of strides with the new axes.
    int8_t expanded_ndim = a.ndim() + axes.ndim();
    int8_t i_axis = 0;
    int8_t i_stride = 0;
    const Strides& strides = a.strides();
    Strides expanded_strides;
    for (int8_t i = 0; i < expanded_ndim; ++i) {
        if (i_axis < axes.ndim() && i == axes[i_axis]) {
            expanded_strides.emplace_back(int64_t{1});
            ++i_axis;
        } else {
            expanded_strides.emplace_back(strides[i_stride]);
            ++i_stride;
        }
    }
    assert(i_axis == axes.ndim());
    assert(i_stride == strides.ndim());
    assert(expanded_strides.ndim() == a.ndim() + axes.ndim());

    return xchainer::internal::MakeArray(
            xchainer::internal::ExpandShape(a.shape(), axes), expanded_strides, a.dtype(), a.device(), a.data(), a.offset());
}

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
        Dtype dtype = x.dtype();

        Array x_mean = xchainer::internal::EmptyReduced(x.shape(), dtype, axis, true, x.device());
        Mean(x, axis, x_mean);

        Array x_var = xchainer::internal::EmptyReduced(x.shape(), dtype, axis, true, x.device());
        Var(x, x_mean, axis, x_var);

        Array out = EmptyLike(x, x.device());
        int64_t n = x.GetTotalSize() / gamma.GetTotalSize();
        VisitFloatingPointDtype(
                dtype, [&x, &x_mean, &x_var, &running_mean, &running_var, &gamma, &beta, eps, decay, &axis, &out, n](auto pt) {
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
                            ExpandDims(gamma, axis).BroadcastTo(out.shape()),
                            ExpandDims(beta, axis).BroadcastTo(out.shape()),
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
                            x_mean,
                            x_var,
                            ExpandDims(running_mean, axis),
                            ExpandDims(running_var, axis));
                });
        return out;
    }

    std::array<Array, 3> Backward(
            const Array& /*x*/, const Array& /*gamma*/, const Array& /*gout*/, Scalar /*eps*/, Scalar /*decay*/, const Axes& /*axis*/)
            override {
        return {Array{}, Array{}, Array{}};
    }
};

}  // namespace

std::shared_ptr<BatchNormForwardBackward> NativeDevice::GetBatchNormForwardBackward() {
    return std::make_shared<NativeBatchNormForwardBackward>();
}

void NativeDevice::Synchronize() {}

}  // namespace native
}  // namespace xchainer
