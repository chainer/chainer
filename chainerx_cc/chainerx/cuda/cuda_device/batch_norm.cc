#include "chainerx/cuda/cuda_device.h"

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include <absl/types/optional.h>

#include <cudnn.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/cudnn.h"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/kernels/normalization.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/normalization.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace cuda {
namespace {

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

// Helper function to update the running mean and running variance.
void UpdateRunning(const Array& running, const Array& running_updated) {
    CHAINERX_ASSERT(running.IsContiguous());
    CHAINERX_ASSERT(running_updated.IsContiguous());
    CHAINERX_ASSERT(&running.device() == &running_updated.device());
    CHAINERX_ASSERT(
            (running.dtype() == running_updated.dtype()) ==
            (internal::GetRawOffsetData(running) == internal::GetRawOffsetData(running_updated)));

    if (running.dtype() == running_updated.dtype()) {
        // Assume that running already holds the updated values.
        return;
    }

    // The running values must be written back.
    const Array& running_casted_back = running_updated.AsType(running.dtype());
    Device& device = running.device();
    device.MemoryCopyFrom(
            internal::GetRawOffsetData(running), internal::GetRawOffsetData(running_casted_back), running.GetNBytes(), device);
}

// Appends singleton axes to make an array with at least 4 dimensions.
// Used for cuDNN BatchNorm, which only supports 4 or 5 dimension input.
Array ExpandToAtLeast4D(const Array& x) {
    if (x.ndim() >= 4) {
        return x;
    }
    Shape shape = x.shape();
    while (shape.size() < 4) {
        shape.push_back(1);
    }
    return x.Reshape(shape);
}

// Derives a secondary tensor descriptor for the batch normalization parameters.
cuda_internal::CudnnTensorDescriptor DeriveBatchNormTensorDescriptor(
        const cuda_internal::CudnnTensorDescriptor& x_desc, cudnnBatchNormMode_t mode) {
    cuda_internal::CudnnTensorDescriptor derive_desc{};
    CheckCudnnError(cudnnDeriveBNTensorDescriptor(*derive_desc, *x_desc, mode));
    return derive_desc;
}

class CudaBatchNormKernel : public BatchNormKernel {
public:
    std::tuple<Array, std::unique_ptr<BatchNormGradState>> Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& running_mean,
            const Array& running_var,
            Scalar eps,
            Scalar decay,
            const Axes& axis,
            bool return_state,
            const absl::optional<Array>& out) override {
        if (CHAINERX_DEBUG) {
            Shape reduced_shape = internal::ReduceShape(x.shape(), axis, true);
            CHAINERX_ASSERT(gamma.shape() == reduced_shape);
            CHAINERX_ASSERT(beta.shape() == reduced_shape);

            int64_t reduced_total_size = reduced_shape.GetTotalSize();
            CHAINERX_ASSERT(running_mean.GetTotalSize() == reduced_total_size);
            CHAINERX_ASSERT(running_var.GetTotalSize() == reduced_total_size);

            CHAINERX_ASSERT(&x.device() == &gamma.device());
            CHAINERX_ASSERT(&x.device() == &beta.device());
            CHAINERX_ASSERT(&x.device() == &running_mean.device());
            CHAINERX_ASSERT(&x.device() == &running_var.device());

            CHAINERX_ASSERT(GetKind(x.dtype()) == DtypeKind::kFloat);
            CHAINERX_ASSERT(GetKind(gamma.dtype()) == DtypeKind::kFloat);
            CHAINERX_ASSERT(GetKind(beta.dtype()) == DtypeKind::kFloat);
            CHAINERX_ASSERT(GetKind(running_mean.dtype()) == DtypeKind::kFloat);
            CHAINERX_ASSERT(GetKind(running_var.dtype()) == DtypeKind::kFloat);
        }
        if (static_cast<double>(eps) < CUDNN_BN_MIN_EPSILON) {
            throw CudnnError{"Minimum allowed epsilon is ", CUDNN_BN_MIN_EPSILON, " but found ", eps, "."};
        }
        if (!running_mean.IsContiguous()) {
            throw DeviceError{"Running mean must to be contiguous for cuDNN to update it in-place."};
        }
        if (!running_var.IsContiguous()) {
            throw DeviceError{"Running variance must to be contiguous for cuDNN to update it in-place."};
        }
        // TODO(hvy): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }

        CudaDevice& device = dynamic_cast<CudaDevice&>(x.device());
        CudaSetDeviceScope scope{device.index()};

        Array x_cont = AsContiguous(x);
        cuda_internal::CudnnTensorDescriptor x_desc{ExpandToAtLeast4D(x_cont)};

        cudnnBatchNormMode_t mode = GetBatchNormMode(axis);
        cuda_internal::CudnnTensorDescriptor gamma_beta_mean_var_desc = DeriveBatchNormTensorDescriptor(x_desc, mode);
        Dtype gamma_beta_mean_var_dtype = gamma_beta_mean_var_desc.GetDtype();

        Array gamma_casted_cont = AsContiguous(gamma, gamma_beta_mean_var_dtype);
        Array beta_casted_cont = AsContiguous(beta, gamma_beta_mean_var_dtype);

        CHAINERX_ASSERT(running_mean.IsContiguous());
        CHAINERX_ASSERT(running_var.IsContiguous());

        // Convert parameter dtypes if they do not match the dtype expected by cuDNN.
        const Array& running_mean_casted =
                running_mean.dtype() != gamma_beta_mean_var_dtype ? running_mean.AsType(gamma_beta_mean_var_dtype) : running_mean;
        const Array& running_var_casted =
                running_var.dtype() != gamma_beta_mean_var_dtype ? running_var.AsType(gamma_beta_mean_var_dtype) : running_var;

        Array x_mean = EmptyLike(gamma_casted_cont, device);
        Array x_inv_std = EmptyLike(gamma_casted_cont, device);

        Array out_cont = EmptyLike(x, x.device());

        Dtype dtype = x.dtype();

        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);

        device_internals.cudnn_handle().Call(
                cudnnBatchNormalizationForwardTraining,
                mode,
                cuda_internal::GetCudnnCoefficientPtr<1>(dtype),
                cuda_internal::GetCudnnCoefficientPtr<0>(dtype),
                *x_desc,
                internal::GetRawOffsetData(x_cont),
                *x_desc,
                internal::GetRawOffsetData(out_cont),
                *gamma_beta_mean_var_desc,
                internal::GetRawOffsetData(gamma_casted_cont),
                internal::GetRawOffsetData(beta_casted_cont),
                1.0 - static_cast<double>(decay),
                internal::GetRawOffsetData(running_mean_casted),
                internal::GetRawOffsetData(running_var_casted),
                static_cast<double>(eps),
                internal::GetRawOffsetData(x_mean),
                internal::GetRawOffsetData(x_inv_std));

        // When data type of parameters is converted, say, from fp16
        // to fp32, the values of fp32 arrays of running_mean and
        // running_var updated by batchNormalizationForwardTraining
        // must be explicitly written back to their original fp16 arrays.
        UpdateRunning(running_mean, running_mean_casted);
        UpdateRunning(running_var, running_var_casted);

        std::unique_ptr<BatchNormGradState> state =
                return_state
                        ? std::make_unique<CudaBatchNormGradState>(std::move(x_cont), std::move(x_mean), std::move(x_inv_std), beta.dtype())
                        : nullptr;

        return std::make_tuple(std::move(out_cont), std::move(state));
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(BatchNormKernel, CudaBatchNormKernel);

class CudaBatchNormGradKernel : public BatchNormGradKernel {
public:
    std::tuple<Array, Array, Array> Call(
            const Array& x,
            const Array& gamma,
            const Array& gout,
            Scalar eps,
            const Axes& axis,
            const std::shared_ptr<BatchNormGradState>& state,
            const absl::optional<Array>& gx,
            const absl::optional<Array>& ggamma,
            const absl::optional<Array>& gbeta) override {
        CHAINERX_ASSERT(gamma.shape() == internal::ReduceShape(x.shape(), axis, true));
        CHAINERX_ASSERT(x.dtype() == gout.dtype());
        CHAINERX_ASSERT(x.shape() == gout.shape());
        CHAINERX_ASSERT(&x.device() == &gamma.device());
        CHAINERX_ASSERT(&x.device() == &gout.device());

        if (static_cast<double>(eps) < CUDNN_BN_MIN_EPSILON) {
            throw CudnnError{"Minimum allowed epsilon is ", CUDNN_BN_MIN_EPSILON, " but found ", eps, "."};
        }
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

        // TODO(hvy): Implement recomputation of x_cont, x_mean and x_inv_std in case they are not given by the state.
        CHAINERX_ASSERT(state != nullptr);
        auto& cuda_state = dynamic_cast<CudaBatchNormGradState&>(*state);
        const Array& x_cont = cuda_state.x_cont();
        const Array& x_mean = cuda_state.x_mean();
        const Array& x_inv_std = cuda_state.x_inv_std();
        Dtype beta_dtype = cuda_state.beta_dtype();
        CHAINERX_ASSERT(x_cont.IsContiguous());
        CHAINERX_ASSERT(x_cont.shape() == x.shape());
        CHAINERX_ASSERT(x_mean.shape() == gamma.shape());
        CHAINERX_ASSERT(x_inv_std.shape() == gamma.shape());
        CHAINERX_ASSERT(&x_cont.device() == &x.device());
        CHAINERX_ASSERT(&x_cont.device() == &x_mean.device());
        CHAINERX_ASSERT(&x_cont.device() == &x_inv_std.device());

        CudaDevice& device = dynamic_cast<CudaDevice&>(x.device());
        CudaSetDeviceScope scope{device.index()};

        Array gout_cont = AsContiguous(gout);
        Array actual_gx = EmptyLike(x, device);
        cuda_internal::CudnnTensorDescriptor x_desc{ExpandToAtLeast4D(x_cont)};

        cudnnBatchNormMode_t mode = GetBatchNormMode(axis);

        cuda_internal::CudnnTensorDescriptor gamma_beta_mean_var_desc = DeriveBatchNormTensorDescriptor(x_desc, mode);
        Dtype gamma_beta_mean_var_dtype = gamma_beta_mean_var_desc.GetDtype();
        Shape gamma_beta_mean_var_shape = internal::ReduceShape(x_cont.shape(), axis, true);

        Array gamma_casted_cont = AsContiguous(gamma, gamma_beta_mean_var_dtype);
        Array actual_ggamma = Empty(gamma_beta_mean_var_shape, gamma_beta_mean_var_dtype, device);
        Array actual_gbeta = Empty(gamma_beta_mean_var_shape, gamma_beta_mean_var_dtype, device);

        CHAINERX_ASSERT(gamma_beta_mean_var_dtype == x_mean.dtype());
        CHAINERX_ASSERT(gamma_beta_mean_var_dtype == x_inv_std.dtype());
        CHAINERX_ASSERT(x_mean.IsContiguous());
        CHAINERX_ASSERT(x_inv_std.IsContiguous());

        Dtype dtype = x_cont.dtype();

        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);

        device_internals.cudnn_handle().Call(
                cudnnBatchNormalizationBackward,
                mode,
                cuda_internal::GetCudnnCoefficientPtr<1>(dtype),
                cuda_internal::GetCudnnCoefficientPtr<0>(dtype),
                cuda_internal::GetCudnnCoefficientPtr<1>(dtype),
                cuda_internal::GetCudnnCoefficientPtr<0>(dtype),
                *x_desc,
                internal::GetRawOffsetData(x_cont),
                *x_desc,
                internal::GetRawOffsetData(gout_cont),
                *x_desc,
                internal::GetRawOffsetData(actual_gx),
                *gamma_beta_mean_var_desc,
                internal::GetRawOffsetData(gamma_casted_cont),
                internal::GetRawOffsetData(actual_ggamma),
                internal::GetRawOffsetData(actual_gbeta),
                static_cast<double>(eps),
                internal::GetRawOffsetData(x_mean),
                internal::GetRawOffsetData(x_inv_std));

        if (actual_ggamma.dtype() != gamma.dtype()) {
            actual_ggamma = actual_ggamma.AsType(gamma.dtype());
        }
        if (actual_gbeta.dtype() != beta_dtype) {
            actual_gbeta = actual_gbeta.AsType(beta_dtype);
        }

        return std::make_tuple(std::move(actual_gx), std::move(actual_ggamma), std::move(actual_gbeta));
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(BatchNormGradKernel, CudaBatchNormGradKernel);

class CudaFixedBatchNormKernel : public FixedBatchNormKernel {
public:
    Array Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& mean,
            const Array& var,
            Scalar eps,
            const Axes& axis,
            const absl::optional<Array>& out) override {
        if (CHAINERX_DEBUG) {
            Shape reduced_shape = internal::ReduceShape(x.shape(), axis, true);
            CHAINERX_ASSERT(gamma.shape() == reduced_shape);
            CHAINERX_ASSERT(beta.shape() == reduced_shape);
            CHAINERX_ASSERT(mean.shape() == reduced_shape);
            CHAINERX_ASSERT(var.shape() == reduced_shape);

            CHAINERX_ASSERT(&x.device() == &gamma.device());
            CHAINERX_ASSERT(&x.device() == &beta.device());
            CHAINERX_ASSERT(&x.device() == &mean.device());
            CHAINERX_ASSERT(&x.device() == &var.device());

            CHAINERX_ASSERT(GetKind(x.dtype()) == DtypeKind::kFloat);
            CHAINERX_ASSERT(GetKind(gamma.dtype()) == DtypeKind::kFloat);
            CHAINERX_ASSERT(GetKind(beta.dtype()) == DtypeKind::kFloat);
            CHAINERX_ASSERT(GetKind(mean.dtype()) == DtypeKind::kFloat);
            CHAINERX_ASSERT(GetKind(var.dtype()) == DtypeKind::kFloat);
        }

        if (static_cast<double>(eps) < CUDNN_BN_MIN_EPSILON) {
            throw CudnnError{"Minimum allowed epsilon is ", CUDNN_BN_MIN_EPSILON, " but found ", eps, "."};
        }
        // TODO(hvy): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }

        CudaDevice& device = dynamic_cast<CudaDevice&>(x.device());
        CudaSetDeviceScope scope{device.index()};

        Array x_cont = AsContiguous(x);
        cuda_internal::CudnnTensorDescriptor x_desc{ExpandToAtLeast4D(x_cont)};

        cudnnBatchNormMode_t mode = GetBatchNormMode(axis);

        cuda_internal::CudnnTensorDescriptor gamma_beta_mean_var_desc = DeriveBatchNormTensorDescriptor(x_desc, mode);
        Dtype gamma_beta_mean_var_dtype = gamma_beta_mean_var_desc.GetDtype();

        Array gamma_casted_cont = AsContiguous(gamma, gamma_beta_mean_var_dtype);
        Array beta_casted_cont = AsContiguous(beta, gamma_beta_mean_var_dtype);
        Array mean_casted_cont = AsContiguous(mean, gamma_beta_mean_var_dtype);
        Array var_casted_cont = AsContiguous(var, gamma_beta_mean_var_dtype);

        Dtype dtype = x_cont.dtype();

        Array actual_out = EmptyLike(x, device);

        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);

        device_internals.cudnn_handle().Call(
                cudnnBatchNormalizationForwardInference,
                GetBatchNormMode(axis),
                cuda_internal::GetCudnnCoefficientPtr<1>(dtype),
                cuda_internal::GetCudnnCoefficientPtr<0>(dtype),
                *x_desc,
                internal::GetRawOffsetData(x_cont),
                *x_desc,
                internal::GetRawOffsetData(actual_out),
                *gamma_beta_mean_var_desc,
                internal::GetRawOffsetData(gamma_casted_cont),
                internal::GetRawOffsetData(beta_casted_cont),
                internal::GetRawOffsetData(mean_casted_cont),
                internal::GetRawOffsetData(var_casted_cont),
                static_cast<double>(eps));

        return actual_out;
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(FixedBatchNormKernel, CudaFixedBatchNormKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
