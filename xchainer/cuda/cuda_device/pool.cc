#include "xchainer/cuda/cuda_device.h"

#include <cassert>
#include <cstdint>
#include <memory>

#include <cudnn.h>

#include "xchainer/array.h"
#include "xchainer/backend_util.h"
#include "xchainer/constant.h"
#include "xchainer/cuda/cudnn.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/pooling.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace cuda {
namespace {

class PoolImpl {
public:
    PoolImpl(
            cudnnHandle_t cudnn_handle,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all,
            cudnnPoolingMode_t cudnn_pooling_mode)
        : cudnn_handle_{cudnn_handle},
          kernel_size_{kernel_size},
          stride_{stride},
          pad_{pad},
          cover_all_{cover_all},
          cudnn_pooling_mode_{cudnn_pooling_mode} {
        if (cover_all_) {
            throw XchainerError{"CUDA pooling does not support cover_all"};
        }
    }

    Array Forward(const Array& x) {
        int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions
        if (ndim < 2) {
            throw DimensionError{"CUDA pooling requires number of spatial dimensions to be greater than or equal to 2"};
        }

        assert(kernel_size_.size() == static_cast<size_t>(ndim));
        assert(stride_.size() == static_cast<size_t>(ndim));
        assert(pad_.size() == static_cast<size_t>(ndim));

        // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
        Shape out_shape{x.shape()[0], x.shape()[1]};
        for (int8_t i = 0; i < ndim; ++i) {
            out_shape.emplace_back(xchainer::internal::GetConvOutDim(x.shape()[i + 2], kernel_size_[i], stride_[i], pad_[i], cover_all_));
            assert(out_shape.back() > 0);
        }

        Array y = Empty(out_shape, x.dtype(), x.device());
        Array x_cont = AsContiguousArray(x);

        internal::CudnnTensorDescriptor x_desc{x_cont};
        internal::CudnnTensorDescriptor y_desc{y};

        internal::CudnnPoolingDescriptor pool_desc{cudnn_pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, kernel_size_, pad_, stride_};

        CheckCudnnError(cudnnPoolingForward(
                cudnn_handle_,
                *pool_desc,
                internal::GetValuePtr<1>(x.dtype()),
                *x_desc,
                xchainer::internal::GetRawOffsetData<void>(x_cont),
                internal::GetValuePtr<0>(x.dtype()),
                *y_desc,
                xchainer::internal::GetRawOffsetData<void>(y)));

        x_ = x.AsConstant();
        y_ = y.AsConstant();

        return y;
    }

    Array Backward(const Array& gout) {
        int8_t ndim = x_.ndim() - 2;  // Number of spacial dimensions
        if (ndim < 2) {
            throw DimensionError{"CUDA pooling requires number of spatial dimensions to be greater than or equal to 2"};
        }

        assert(kernel_size_.size() == static_cast<size_t>(ndim));
        assert(stride_.size() == static_cast<size_t>(ndim));
        assert(pad_.size() == static_cast<size_t>(ndim));
        assert(gout.shape() == y_.shape());

        Array gx = EmptyLike(x_, x_.device());
        Array y_cont = AsContiguousArray(y_);
        Array gout_cont = AsContiguousArray(gout);
        Array x_cont = AsContiguousArray(x_);

        internal::CudnnTensorDescriptor y_desc{y_cont};
        internal::CudnnTensorDescriptor gout_desc{gout_cont};
        internal::CudnnTensorDescriptor x_desc{x_cont};
        internal::CudnnTensorDescriptor gx_desc{gx};

        internal::CudnnPoolingDescriptor pool_desc{cudnn_pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, kernel_size_, pad_, stride_};

        CheckCudnnError(cudnnPoolingBackward(
                cudnn_handle_,
                *pool_desc,
                internal::GetValuePtr<1>(x_.dtype()),
                *y_desc,
                xchainer::internal::GetRawOffsetData<void>(y_cont),
                *gout_desc,
                xchainer::internal::GetRawOffsetData<void>(gout_cont),
                *x_desc,
                xchainer::internal::GetRawOffsetData<void>(x_cont),
                internal::GetValuePtr<0>(x_.dtype()),
                *gx_desc,
                xchainer::internal::GetRawOffsetData<void>(gx)));

        return gx;
    }

private:
    cudnnHandle_t cudnn_handle_;
    const StackVector<int64_t, kMaxNdim> kernel_size_;
    const StackVector<int64_t, kMaxNdim> stride_;
    const StackVector<int64_t, kMaxNdim> pad_;
    bool cover_all_;
    cudnnPoolingMode_t cudnn_pooling_mode_;
    Array x_;
    Array y_;
};

class CudaMaxPoolForwardBackward : public xchainer::MaxPoolForwardBackward {
public:
    explicit CudaMaxPoolForwardBackward(
            cudnnHandle_t cudnn_handle,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all)
        : pool_impl_{cudnn_handle, kernel_size, stride, pad, cover_all, CUDNN_POOLING_MAX} {}

    Array Forward(const Array& x) override { return pool_impl_.Forward(x); }

    Array Backward(const Array& gout) override { return pool_impl_.Backward(gout); }

    // TODO(hvy): Implement me.
    Array DoubleBackward(const Array& /*ggx*/) override { return Array{}; }

private:
    PoolImpl pool_impl_;
};

}  // namespace

std::unique_ptr<MaxPoolForwardBackward> CudaDevice::GetMaxPoolForwardBackward(
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    return std::make_unique<CudaMaxPoolForwardBackward>(cudnn_handle(), kernel_size, stride, pad, cover_all);
}

namespace {

cudnnPoolingMode_t GetCudnnPoolingMode(AveragePoolPadMode pad_mode) {
    switch (pad_mode) {
        case AveragePoolPadMode::kZero:
            return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        case AveragePoolPadMode::kIgnore:
            return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        default:
            XCHAINER_NEVER_REACH();
    }
}

class CudaAveragePoolForwardBackward : public xchainer::AveragePoolForwardBackward {
public:
    explicit CudaAveragePoolForwardBackward(
            cudnnHandle_t cudnn_handle,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            AveragePoolPadMode pad_mode)
        : pool_impl_{cudnn_handle, kernel_size, stride, pad, false, GetCudnnPoolingMode(pad_mode)} {}

    Array Forward(const Array& x) override { return pool_impl_.Forward(x); }

    Array Backward(const Array& gout) override { return pool_impl_.Backward(gout); }

private:
    PoolImpl pool_impl_;
};

}  // namespace

std::unique_ptr<AveragePoolForwardBackward> CudaDevice::GetAveragePoolForwardBackward(
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        AveragePoolPadMode pad_mode) {
    return std::make_unique<CudaAveragePoolForwardBackward>(cudnn_handle(), kernel_size, stride, pad, pad_mode);
}

}  // namespace cuda
}  // namespace xchainer
