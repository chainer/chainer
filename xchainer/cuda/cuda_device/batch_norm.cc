#include "xchainer/cuda/cuda_device.h"

#include <cassert>
#include <cstdint>
#include <memory>

#include <cudnn.h>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/cuda/cudnn.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/routines/creation.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace cuda {
namespace {

class CudaBatchNormForwardBackward : public xchainer::BatchNormForwardBackward {
public:
    explicit CudaBatchNormForwardBackward(internal::CudnnContext& cudnn_context) : cudnn_context_{cudnn_context} {}

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
        if (!running_mean.IsContiguous()) {
            throw DeviceError{"Running mean must to be contiguous for cuDNN to update it in-place."};
        }
        if (!running_var.IsContiguous()) {
            throw DeviceError{"Running variance must to be contiguous for cuDNN to update it in-place."};
        }

        Device& device = x.device();

        // Initialize cache.
        result_mean_ = EmptyLike(gamma, device);
        result_inv_var_ = EmptyLike(gamma, device);

        Array out = EmptyLike(x, device);
        cudnn_context_.BatchNormalizationForwardTraining(
                GetBatchNormMode(axis),
                x,
                out,
                gamma,
                beta,
                1.0 - static_cast<double>(decay),
                running_mean,
                running_var,
                static_cast<double>(eps),
                result_mean_,
                result_inv_var_);
        return out;
    }

    // TODO(hvy): Implement me.
    std::array<Array, 3> Backward(
            const Array& /*x*/, const Array& /*gamma*/, const Array& /*gout*/, Scalar /*eps*/, const Axes& /*axis*/) override {
        return {Array{}, Array{}, Array{}};
    }

    // TODO(niboshi): Implement me.
    std::array<Array, 3> DoubleBackward(const Array& /*ggx*/, const Array& /*gggamma*/, const Array& /*ggbeta*/) {
        return {Array{}, Array{}, Array{}};
    }

private:
    cudnnBatchNormMode_t GetBatchNormMode(const Axes& axis) {
        if (axis.ndim() == 1 && axis[0] == 0) {  // (1, channels, (depth, )height, width)
            return CUDNN_BATCHNORM_PER_ACTIVATION;
        }
        if ((axis.ndim() == 3 && axis[0] == 0 && axis[1] == 2 && axis[2] == 3) ||
            (axis.ndim() == 4 && axis[0] == 0 && axis[1] == 2 && axis[2] == 3 && axis[3] == 4)) {  // (1, channels, (1, )1, 1)
            // TODO(hvy): Consider CUDNN_BATCHNORM_SPATIAL_PERSISTENT if we can afford to check for overflow, with or without blocking.
            return CUDNN_BATCHNORM_SPATIAL;
        }
        throw DimensionError{"Invalid axis for BatchNorm using cuDNN ", axis, ". Expected 1, 3 or 4 dimensions."};
    }

    internal::CudnnContext& cudnn_context_;

    // Cache intermediate results during Forward for reuse in Backward.
    Array result_mean_{};
    Array result_inv_var_{};
};

}  // namespace

std::unique_ptr<BatchNormForwardBackward> CudaDevice::GetBatchNormForwardBackward() {
    return std::make_unique<CudaBatchNormForwardBackward>(cudnn_context_);
}

}  // namespace cuda
}  // namespace xchainer
