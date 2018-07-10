#include "xchainer/cuda/cuda_device.h"

#include <cassert>
#include <cstdint>
#include <memory>

#include <cudnn.h>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/backend_util.h"
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

// TODO(sonots): Support other than 4, 5 dimensional arrays by reshaping into 4-dimensional arrays as Chainer does.
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

class CudnnBNTensorDescriptor {
public:
    CudnnBNTensorDescriptor(const internal::CudnnTensorDescriptor& x_desc, cudnnBatchNormMode_t mode) : CudnnBNTensorDescriptor{} {
        CheckCudnnError(cudnnDeriveBNTensorDescriptor(desc_, *x_desc, mode));
    }

    CudnnBNTensorDescriptor(const CudnnBNTensorDescriptor&) = delete;
    CudnnBNTensorDescriptor& operator=(const CudnnBNTensorDescriptor&) = delete;

    CudnnBNTensorDescriptor(CudnnBNTensorDescriptor&&) = delete;
    CudnnBNTensorDescriptor& operator=(CudnnBNTensorDescriptor&&) = delete;

    ~CudnnBNTensorDescriptor() {
        if (desc_ != nullptr) {
            CheckCudnnError(cudnnDestroyTensorDescriptor(desc_));
        }
    }

    cudnnTensorDescriptor_t descriptor() const { return desc_; }
    cudnnTensorDescriptor_t operator*() const { return desc_; }

    Dtype GetDtype() {
        cudnnDataType_t cudnn_dtype{};
        int ndim{};

        CheckCudnnError(cudnnGetTensorNdDescriptor(desc_, 0, &cudnn_dtype, &ndim, nullptr, nullptr));

        switch (cudnn_dtype) {
            case CUDNN_DATA_DOUBLE:
                return Dtype::kFloat64;
            case CUDNN_DATA_FLOAT:
                return Dtype::kFloat32;
            // TODO(sonots): Support float16
            // case CUDNN_DATA_HALF;
            //     return Dtype::kFloat16;
            default:
                throw DtypeError{"Unsupported cudnn data type: ", cudnn_dtype};
        }
    }

private:
    CudnnBNTensorDescriptor() { CheckCudnnError(cudnnCreateTensorDescriptor(&desc_)); }
    cudnnTensorDescriptor_t desc_{};
};

class CudaBatchNormForwardBackward : public xchainer::GenericBatchNormForwardBackward {
public:
    explicit CudaBatchNormForwardBackward(
            cudnnHandle_t cudnn_handle, const Array& running_mean, const Array& running_var, Scalar eps, Scalar decay, const Axes& axis)
        : GenericBatchNormForwardBackward{running_mean, running_var, eps, decay, axis}, cudnn_handle_{cudnn_handle} {
        if (static_cast<double>(eps) < CUDNN_BN_MIN_EPSILON) {
            throw CudnnError{"Minimum allowed epsilon is ", CUDNN_BN_MIN_EPSILON, " but found ", eps, "."};
        }
        if (!running_mean.IsContiguous()) {
            throw DeviceError{"Running mean must to be contiguous for cuDNN to update it in-place."};
        }
        if (!running_var.IsContiguous()) {
            throw DeviceError{"Running variance must to be contiguous for cuDNN to update it in-place."};
        }
    }

    Array Forward(const Array& x, const Array& gamma, const Array& beta) override {
#ifndef NDEBUG
        {
            Shape reduced_shape = xchainer::internal::ReduceShape(x.shape(), axis(), true);
            assert(gamma.shape() == reduced_shape);
            assert(beta.shape() == reduced_shape);

            int64_t reduced_total_size = reduced_shape.GetTotalSize();
            assert(running_mean().GetTotalSize() == reduced_total_size);
            assert(running_var().GetTotalSize() == reduced_total_size);

            assert(&x.device() == &gamma.device());
            assert(&x.device() == &beta.device());
            assert(&x.device() == &running_mean().device());
            assert(&x.device() == &running_var().device());

            assert(x.dtype() == gamma.dtype());
            assert(x.dtype() == beta.dtype());
            assert(x.dtype() == running_mean().dtype());
            assert(x.dtype() == running_var().dtype());

            assert(!xchainer::internal::HasAnyArrayNode(x));
            assert(!xchainer::internal::HasAnyArrayNode(gamma));
            assert(!xchainer::internal::HasAnyArrayNode(beta));
        }
#endif  // NDEBUG

        Device& device = x.device();
        Dtype dtype = x.dtype();

        Array x_cont = AsContiguousArray(x);
        internal::CudnnTensorDescriptor x_desc{x_cont};
        cudnnBatchNormMode_t mode = GetBatchNormMode(axis());

        CudnnBNTensorDescriptor gamma_beta_mean_var_desc{x_desc, mode};
        Dtype gamma_beta_mean_var_dtype = gamma_beta_mean_var_desc.GetDtype();

        Array gamma_casted_cont = AsContiguousArray(gamma.AsType(gamma_beta_mean_var_dtype, false));
        Array beta_casted_cont = AsContiguousArray(beta.AsType(gamma_beta_mean_var_dtype, false));

        assert(running_mean().IsContiguous());
        assert(running_var().IsContiguous());
        Array running_mean_casted = running_mean().AsType(gamma_beta_mean_var_dtype, false);
        Array running_var_casted = running_var().AsType(gamma_beta_mean_var_dtype, false);

        Array out = EmptyLike(x, device);
        Array x_mean = EmptyLike(gamma_casted_cont, device);
        Array x_inv_std = EmptyLike(gamma_casted_cont, device);

        CheckCudnnError(cudnnBatchNormalizationForwardTraining(
                cudnn_handle_,
                mode,
                internal::GetValuePtr<1>(dtype),
                internal::GetValuePtr<0>(dtype),
                *x_desc,
                xchainer::internal::GetRawOffsetData<void>(x_cont),
                *x_desc,
                xchainer::internal::GetRawOffsetData<void>(out),
                *gamma_beta_mean_var_desc,
                xchainer::internal::GetRawOffsetData<void>(gamma_casted_cont),
                xchainer::internal::GetRawOffsetData<void>(beta_casted_cont),
                1.0 - static_cast<double>(decay()),
                xchainer::internal::GetRawOffsetData<void>(running_mean_casted),
                xchainer::internal::GetRawOffsetData<void>(running_var_casted),
                static_cast<double>(eps()),
                xchainer::internal::GetRawOffsetData<void>(x_mean),
                xchainer::internal::GetRawOffsetData<void>(x_inv_std)));

        // When data type of prameters is converted, say, from fp16
        // to fp32, the values of fp32 arrays of running_mean and
        // running_var updated by batchNormalizationForwardTraining
        // must be explicitly written back to their original fp16 arrays.
        //
        // TODO(sonots): write tests after we supports fp16
        if (dtype != gamma_beta_mean_var_dtype) {
            assert(running_mean().IsContiguous());
            assert(running_mean_casted.IsContiguous());
            assert(running_var().IsContiguous());
            assert(running_var_casted.IsContiguous());

            Array running_mean_casted_back = running_mean_casted.AsType(dtype, false);
            Array running_var_casted_back = running_var_casted.AsType(dtype, false);

            device.MemoryCopyFrom(
                    xchainer::internal::GetRawOffsetData<void>(running_mean()),
                    xchainer::internal::GetRawOffsetData<void>(running_mean_casted_back),
                    running_mean().GetNBytes(),
                    device);
            device.MemoryCopyFrom(
                    xchainer::internal::GetRawOffsetData<void>(running_var()),
                    xchainer::internal::GetRawOffsetData<void>(running_var_casted_back),
                    running_var().GetNBytes(),
                    device);
        }

        SetForwardResults(x_cont, gamma, x_mean, x_inv_std);

        return out;
    }

    std::array<Array, 3> Backward(const Array& gout) override {
        const Array& x_cont = this->x();
        const Array& gamma = this->gamma();
        const Array& x_mean = this->x_mean();
        const Array& x_inv_std = this->x_inv_std();
#ifndef NDEBUG
        {
            Shape reduced_shape = xchainer::internal::ReduceShape(x_cont.shape(), axis(), true);
            assert(reduced_shape == gamma.shape());
            assert(x_cont.shape() == gout.shape());

            assert(x_mean.body() != nullptr);
            assert(x_inv_std.body() != nullptr);

            assert(&x_cont.device() == &gamma.device());
            assert(&x_cont.device() == &gout.device());
            assert(&x_cont.device() == &x_mean.device());
            assert(&x_cont.device() == &x_inv_std.device());

            assert(x_cont.dtype() == gamma.dtype());
            assert(x_cont.dtype() == gout.dtype());

            assert(x_cont.IsContiguous());

            assert(!xchainer::internal::HasAnyArrayNode(gout));
        }
#endif  // NDEBUG

        Device& device = x_cont.device();
        Dtype dtype = x_cont.dtype();

        Array gout_cont = AsContiguousArray(gout);
        Array gx = EmptyLike(x_cont, device);

        internal::CudnnTensorDescriptor x_desc{x_cont};
        cudnnBatchNormMode_t mode = GetBatchNormMode(axis());

        CudnnBNTensorDescriptor gamma_beta_mean_var_desc{x_desc, mode};
        Dtype gamma_beta_mean_var_dtype = gamma_beta_mean_var_desc.GetDtype();
        Shape gamma_beta_mean_var_shape = xchainer::internal::ReduceShape(x_cont.shape(), axis(), false);

        Array gamma_casted_cont = AsContiguousArray(gamma.AsType(gamma_beta_mean_var_dtype, false));
        Array ggamma = Empty(gamma_beta_mean_var_shape, gamma_beta_mean_var_dtype, device);
        Array gbeta = Empty(gamma_beta_mean_var_shape, gamma_beta_mean_var_dtype, device);
        assert(gamma_beta_mean_var_dtype == x_mean.dtype());
        assert(gamma_beta_mean_var_dtype == x_inv_std.dtype());
        assert(x_mean.IsContiguous());
        assert(x_inv_std.IsContiguous());

        CheckCudnnError(cudnnBatchNormalizationBackward(
                cudnn_handle_,
                mode,
                internal::GetValuePtr<1>(dtype),
                internal::GetValuePtr<0>(dtype),
                internal::GetValuePtr<1>(dtype),
                internal::GetValuePtr<0>(dtype),
                *x_desc,
                xchainer::internal::GetRawOffsetData<void>(x_cont),
                *x_desc,
                xchainer::internal::GetRawOffsetData<void>(gout_cont),
                *x_desc,
                xchainer::internal::GetRawOffsetData<void>(gx),
                *gamma_beta_mean_var_desc,
                xchainer::internal::GetRawOffsetData<void>(gamma_casted_cont),
                xchainer::internal::GetRawOffsetData<void>(ggamma),
                xchainer::internal::GetRawOffsetData<void>(gbeta),
                static_cast<double>(eps()),
                xchainer::internal::GetRawOffsetData<void>(x_mean),
                xchainer::internal::GetRawOffsetData<void>(x_inv_std)));

        if (gamma_beta_mean_var_dtype != dtype) {
            Array ggamma_casted = ggamma.AsType(dtype, false);
            Array gbeta_casted = ggamma.AsType(dtype, false);
            SetBackwardResults(gout_cont, gx, ggamma_casted);
            return {std::move(gx), std::move(ggamma_casted), std::move(gbeta_casted)};
        }
        SetBackwardResults(gout_cont, gx, ggamma);
        return {std::move(gx), std::move(ggamma), std::move(gbeta)};
    }

private:
    cudnnHandle_t cudnn_handle_;
};

}  // namespace

std::unique_ptr<BatchNormForwardBackward> CudaDevice::GetBatchNormForwardBackward(
        const Array& running_mean, const Array& running_var, Scalar eps, Scalar decay, const Axes& axis) {
    return std::make_unique<CudaBatchNormForwardBackward>(cudnn_handle(), running_mean, running_var, eps, decay, axis);
}

Array CudaDevice::FixedBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const Axes& axis) {
    if (static_cast<double>(eps) < CUDNN_BN_MIN_EPSILON) {
        throw CudnnError{"Minimum allowed epsilon is ", CUDNN_BN_MIN_EPSILON, " but found ", eps, "."};
    }

#ifndef NDEBUG
    {
        Shape reduced_shape = xchainer::internal::ReduceShape(x.shape(), axis, true);
        assert(gamma.shape() == reduced_shape);
        assert(beta.shape() == reduced_shape);
        assert(mean.shape() == reduced_shape);
        assert(var.shape() == reduced_shape);

        assert(&x.device() == &gamma.device());
        assert(&x.device() == &beta.device());
        assert(&x.device() == &mean.device());
        assert(&x.device() == &var.device());

        assert(x.dtype() == gamma.dtype());
        assert(x.dtype() == beta.dtype());
        assert(x.dtype() == mean.dtype());
        assert(x.dtype() == var.dtype());
    }
#endif  // NDEBUG

    Array x_cont = AsContiguousArray(x);
    internal::CudnnTensorDescriptor x_desc{x_cont};
    cudnnBatchNormMode_t mode = GetBatchNormMode(axis);

    CudnnBNTensorDescriptor gamma_beta_mean_var_desc{x_desc, mode};
    Dtype gamma_beta_mean_var_dtype = gamma_beta_mean_var_desc.GetDtype();

    Array gamma_casted_cont = AsContiguousArray(gamma.AsType(gamma_beta_mean_var_dtype, false));
    Array beta_casted_cont = AsContiguousArray(beta.AsType(gamma_beta_mean_var_dtype, false));
    Array mean_casted_cont = AsContiguousArray(mean.AsType(gamma_beta_mean_var_dtype, false));
    Array var_casted_cont = AsContiguousArray(var.AsType(gamma_beta_mean_var_dtype, false));

    Array out = EmptyLike(x, x.device());

    CheckCudnnError(cudnnBatchNormalizationForwardInference(
            cudnn_handle(),
            GetBatchNormMode(axis),
            internal::GetValuePtr<1>(x.dtype()),
            internal::GetValuePtr<0>(x.dtype()),
            *x_desc,
            xchainer::internal::GetRawOffsetData<void>(x_cont),
            *x_desc,
            xchainer::internal::GetRawOffsetData<void>(out),
            *gamma_beta_mean_var_desc,
            xchainer::internal::GetRawOffsetData<void>(gamma_casted_cont),
            xchainer::internal::GetRawOffsetData<void>(beta_casted_cont),
            xchainer::internal::GetRawOffsetData<void>(mean_casted_cont),
            xchainer::internal::GetRawOffsetData<void>(var_casted_cont),
            static_cast<double>(eps)));

    return out;
}

}  // namespace cuda
}  // namespace xchainer
