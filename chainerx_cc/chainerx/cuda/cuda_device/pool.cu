#include "chainerx/cuda/cuda_device.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include <cudnn.h>

#include "chainerx/array.h"
#include "chainerx/backend_util.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/cudnn.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/macro.h"
#include "chainerx/numeric_limits.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/pooling.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {
namespace {

// Struct that allows passing StackVectors to CUDA kernels.
struct CudaStackVector {
    explicit CudaStackVector(const StackVector<int64_t, kMaxNdim>& stack_vector) {
        std::copy_n(stack_vector.begin(), stack_vector.size(), data);
    }
    int64_t data[kMaxNdim];
};

// Uses the previously computed y to find the indices for which the upstream gradients should be propagated.
// It is faster than looking for the argmax again since we only have to do a single comparison.
// TODO(hvy): Make the spatial dimensionality a template parameter to allow unrolling the loops.
template <typename T>
__global__ void MaxPoolDoubleBackwardKernel(
        IndexableArray<const T> ggx_iarray,
        IndexableArray<const T> x_iarray,
        IndexableArray<const T> y_iarray,
        IndexableArray<T> ggy_iarray,
        Indexer<> x_indexer,
        Indexer<> y_indexer,
        Indexer<> kernel_indexer,
        CudaStackVector stride,
        CudaStackVector pad) {
    auto it_kernel = kernel_indexer.It(kernel_indexer.total_size() - 1);
    auto it_x = x_indexer.It(0);

    for (auto it_y = y_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); it_y; ++it_y) {
        it_x.index()[0] = it_y.index()[0];  // batch.
        it_x.index()[1] = it_y.index()[1];  // channel.

        cuda_internal::StorageType<T> y = y_iarray[it_y];

        // Iterate over the kernel in the reverse order, since the resulting index should the be first match.
        for (it_kernel.Restart(); it_kernel.raw_index() >= 0; --it_kernel) {
            for (int8_t i = 2; i < x_indexer.ndim(); ++i) {
                int64_t idx = it_y.index()[i] * stride.data[i - 2] - pad.data[i - 2] + it_kernel.index()[i - 2];
                idx = max(idx, int64_t{0});
                idx = min(idx, x_indexer.shape()[i] - 1);
                it_x.index()[i] = idx;
            }
            if (y == x_iarray[it_x]) {
                ggy_iarray[it_y] = ggx_iarray[it_x];
            }
        }
    }
}

class PoolImpl {
public:
    PoolImpl(
            cuda_internal::CudnnHandle& cudnn_handle,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            bool cover_all,
            cudnnPoolingMode_t cudnn_pooling_mode)
        : cudnn_handle_{cudnn_handle},
          kernel_size_{std::move(kernel_size)},
          stride_{std::move(stride)},
          pad_{std::move(pad)},
          cover_all_{cover_all},
          cudnn_pooling_mode_{cudnn_pooling_mode} {}

    Array Forward(const Array& x) {
        int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions
        if (ndim != 2 && ndim != 3) {
            throw DimensionError{"ChainerX cuDNN pooling supports only 2 and 3 spatial dimensions."};
        }

        CHAINERX_ASSERT(kernel_size_.size() == static_cast<size_t>(ndim));
        CHAINERX_ASSERT(stride_.size() == static_cast<size_t>(ndim));
        CHAINERX_ASSERT(pad_.size() == static_cast<size_t>(ndim));

        // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
        Shape out_shape{x.shape()[0], x.shape()[1]};
        for (int8_t i = 0; i < ndim; ++i) {
            out_shape.emplace_back(internal::GetConvOutDim(x.shape()[i + 2], kernel_size_[i], stride_[i], pad_[i], cover_all_));
            CHAINERX_ASSERT(out_shape.back() > 0);
        }

        Device& device = x.device();
        Dtype dtype = x.dtype();

        CudaSetDeviceScope scope{device.index()};

        Array y = Empty(out_shape, dtype, device);
        Array x_cont = AsContiguousArray(x);

        cuda_internal::CudnnTensorDescriptor x_desc{x_cont};
        cuda_internal::CudnnTensorDescriptor y_desc{y};

        cuda_internal::CudnnPoolingDescriptor pool_desc{cudnn_pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, kernel_size_, pad_, stride_};

        cudnn_handle_.Call(
                cudnnPoolingForward,
                *pool_desc,
                cuda_internal::GetCudnnCoefficientPtr<1>(dtype),
                *x_desc,
                internal::GetRawOffsetData(x_cont),
                cuda_internal::GetCudnnCoefficientPtr<0>(dtype),
                *y_desc,
                internal::GetRawOffsetData(y));

        x_ = x;
        y_ = y;

        return y;
    }

    Array Backward(const Array& gout) {
        int8_t ndim = x_.ndim() - 2;  // Number of spacial dimensions
        if (ndim < 2) {
            throw DimensionError{"CUDA pooling requires number of spatial dimensions to be greater than or equal to 2"};
        }

        CHAINERX_ASSERT(kernel_size_.size() == static_cast<size_t>(ndim));
        CHAINERX_ASSERT(stride_.size() == static_cast<size_t>(ndim));
        CHAINERX_ASSERT(pad_.size() == static_cast<size_t>(ndim));
        CHAINERX_ASSERT(gout.shape() == y_.shape());

        Device& device = x_.device();
        Dtype dtype = x_.dtype();

        CudaSetDeviceScope scope{device.index()};

        Array gx = EmptyLike(x_, device);
        Array y_cont = AsContiguousArray(y_);
        Array gout_cont = AsContiguousArray(gout);
        Array x_cont = AsContiguousArray(x_);

        cuda_internal::CudnnTensorDescriptor y_desc{y_cont};
        cuda_internal::CudnnTensorDescriptor gout_desc{gout_cont};
        cuda_internal::CudnnTensorDescriptor x_desc{x_cont};
        cuda_internal::CudnnTensorDescriptor gx_desc{gx};

        cuda_internal::CudnnPoolingDescriptor pool_desc{cudnn_pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, kernel_size_, pad_, stride_};

        cudnn_handle_.Call(
                cudnnPoolingBackward,
                *pool_desc,
                cuda_internal::GetCudnnCoefficientPtr<1>(dtype),
                *y_desc,
                internal::GetRawOffsetData(y_cont),
                *gout_desc,
                internal::GetRawOffsetData(gout_cont),
                *x_desc,
                internal::GetRawOffsetData(x_cont),
                cuda_internal::GetCudnnCoefficientPtr<0>(dtype),
                *gx_desc,
                internal::GetRawOffsetData(gx));

        return gx;
    }

    Array DoubleBackward(const Array& ggx) {
        Device& device = ggx.device();
        CudaSetDeviceScope scope{device.index()};

        Array ggy = EmptyLike(y_, device);

        VisitFloatingPointDtype(ggy.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;

            IndexableArray<const T> ggx_iarray{ggx};
            IndexableArray<const T> x_iarray{x_};
            IndexableArray<const T> y_iarray{y_};
            IndexableArray<T> ggy_iarray{ggy};

            Indexer<> x_indexer{x_.shape()};
            Indexer<> y_indexer{y_.shape()};
            Indexer<> kernel_indexer{Shape{kernel_size_.begin(), kernel_size_.end()}};

            static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&MaxPoolDoubleBackwardKernel<T>).block_size;
            int64_t total_size = y_indexer.total_size();
            int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
            int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

            MaxPoolDoubleBackwardKernel<<<grid_size, block_size>>>(
                    ggx_iarray,
                    x_iarray,
                    y_iarray,
                    ggy_iarray,
                    x_indexer,
                    y_indexer,
                    kernel_indexer,
                    CudaStackVector{stride_},
                    CudaStackVector{pad_});
        });

        return ggy;
    }

private:
    cuda_internal::CudnnHandle& cudnn_handle_;
    const StackVector<int64_t, kMaxNdim> kernel_size_;
    const StackVector<int64_t, kMaxNdim> stride_;
    const StackVector<int64_t, kMaxNdim> pad_;
    bool cover_all_;
    cudnnPoolingMode_t cudnn_pooling_mode_;
    Array x_;
    Array y_;
};

class CudaMaxPoolForwardBackward : public chainerx::MaxPoolForwardBackward {
public:
    explicit CudaMaxPoolForwardBackward(
            cuda_internal::CudnnHandle& cudnn_handle,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all)
        : pool_impl_{cudnn_handle, kernel_size, stride, pad, cover_all, CUDNN_POOLING_MAX} {}

    Array Forward(const Array& x) override { return pool_impl_.Forward(x); }

    Array Backward(const Array& gout) override { return pool_impl_.Backward(gout); }

    Array DoubleBackward(const Array& ggx) override { return pool_impl_.DoubleBackward(ggx); }

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
            CHAINERX_NEVER_REACH();
    }
}

class CudaAveragePoolForwardBackward : public chainerx::AveragePoolForwardBackward {
public:
    explicit CudaAveragePoolForwardBackward(
            cuda_internal::CudnnHandle& cudnn_handle,
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
}  // namespace chainerx
