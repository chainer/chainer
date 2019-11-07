#include "chainerx/cuda/cuda_device.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include <absl/types/optional.h>

#include <cudnn.h>

#include "chainerx/array.h"
#include "chainerx/backend_util.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/cudnn.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/kernels/pooling.h"
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
struct CudaDims {
    explicit CudaDims(const Dims& stack_vector) { std::copy_n(stack_vector.begin(), stack_vector.size(), data); }
    int64_t data[kMaxNdim];
};

// Uses the previously computed out to find the indices for which the upstream gradients should be propagated.
// It is faster than looking for the argmax again since we only have to do a single comparison.
// TODO(hvy): Make the spatial dimensionality a template parameter to allow unrolling the loops.
template <typename T>
__global__ void MaxPoolDoubleBackwardKernel(
        IndexableArray<const T> ggx_iarray,
        IndexableArray<const T> x_iarray,
        IndexableArray<const T> out_iarray,
        IndexableArray<T> ggout_iarray,
        Indexer<> x_indexer,
        Indexer<> out_indexer,
        Indexer<> kernel_indexer,
        CudaDims stride,
        CudaDims pad) {
    auto it_kernel = kernel_indexer.It(kernel_indexer.total_size() - 1);
    auto it_x = x_indexer.It(0);

    int64_t id = static_cast<int64_t>(blockIdx.x);
    int64_t size = static_cast<int64_t>(gridDim.x);
    int64_t block_dim = static_cast<int64_t>(blockDim.x);
    id = id * block_dim + static_cast<int64_t>(threadIdx.x);
    size *= block_dim;
    for (auto it_out = out_indexer.It(id, size); it_out; ++it_out) {
        it_x.index()[0] = it_out.index()[0];  // batch.
        it_x.index()[1] = it_out.index()[1];  // channel.

        cuda_internal::StorageType<T> out = out_iarray[it_out];

        // Iterate over the kernel in the reverse order, since the resulting index should the be first match.
        for (it_kernel.Restart(); it_kernel.raw_index() >= 0; --it_kernel) {
            for (int8_t i = 2; i < x_indexer.ndim(); ++i) {
                int64_t idx = it_out.index()[i] * stride.data[i - 2] - pad.data[i - 2] + it_kernel.index()[i - 2];
                idx = max(idx, int64_t{0});
                idx = min(idx, x_indexer.shape()[i] - 1);
                it_x.index()[i] = idx;
            }
            if (out == x_iarray[it_x]) {
                ggout_iarray[it_out] = ggx_iarray[it_x];
            }
        }
    }
}

Array Pool(
        cudnnPoolingMode_t cudnn_pooling_mode,
        const Array& x,
        Dims kernel_size,
        Dims stride,
        Dims pad,
        bool cover_all,
        const absl::optional<Array>& out) {
    CHAINERX_ASSERT(kernel_size.size() == static_cast<size_t>(x.ndim() - 2));
    CHAINERX_ASSERT(stride.size() == static_cast<size_t>(x.ndim() - 2));
    CHAINERX_ASSERT(pad.size() == static_cast<size_t>(x.ndim() - 2));

    // TODO(hvy): Implement and test the `out` argument.
    if (out.has_value()) {
        throw NotImplementedError{"Passing out as an argument is not yet supported."};
    }

    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    if (ndim != 2 && ndim != 3) {
        throw DimensionError{"ChainerX cuDNN pooling supports only 2 and 3 spatial dimensions."};
    }

    // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    Shape out_shape{x.shape()[0], x.shape()[1]};
    for (int8_t i = 0; i < ndim; ++i) {
        out_shape.emplace_back(internal::GetConvOutDim(x.shape()[i + 2], kernel_size[i], stride[i], pad[i], cover_all));
        CHAINERX_ASSERT(out_shape.back() > 0);
    }

    CudaDevice& device = dynamic_cast<CudaDevice&>(x.device());
    Dtype dtype = x.dtype();

    CudaSetDeviceScope scope{device.index()};

    Array actual_out = Empty(out_shape, dtype, device);
    Array x_cont = AsContiguousArray(x);

    cuda_internal::CudnnTensorDescriptor x_desc{x_cont};
    cuda_internal::CudnnTensorDescriptor out_desc{actual_out};

    cuda_internal::CudnnPoolingDescriptor pool_desc{cudnn_pooling_mode, CUDNN_NOT_PROPAGATE_NAN, kernel_size, pad, stride};

    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);

    device_internals.cudnn_handle().Call(
            cudnnPoolingForward,
            *pool_desc,
            cuda_internal::GetCudnnCoefficientPtr<1>(dtype),
            *x_desc,
            internal::GetRawOffsetData(x_cont),
            cuda_internal::GetCudnnCoefficientPtr<0>(dtype),
            *out_desc,
            internal::GetRawOffsetData(actual_out));

    return actual_out;
}

Array PoolGrad(
        cudnnPoolingMode_t cudnn_pooling_mode,
        const Array& x,
        const Array& out,
        const Array& gout,
        Dims kernel_size,
        Dims stride,
        Dims pad,
        const absl::optional<Array>& gx) {
    CHAINERX_ASSERT(out.shape() == gout.shape());
    CHAINERX_ASSERT(kernel_size.size() == static_cast<size_t>(x.ndim() - 2));
    CHAINERX_ASSERT(stride.size() == static_cast<size_t>(x.ndim() - 2));
    CHAINERX_ASSERT(pad.size() == static_cast<size_t>(x.ndim() - 2));

    // TODO(hvy): Implement and test the `gx` argument.
    if (gx.has_value()) {
        throw NotImplementedError{"Passing gx as an argument is not yet supported."};
    }

    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    if (ndim < 2) {
        throw DimensionError{"CUDA pooling requires number of spatial dimensions to be greater than or equal to 2"};
    }

    CudaDevice& device = dynamic_cast<CudaDevice&>(x.device());
    Dtype dtype = x.dtype();

    CudaSetDeviceScope scope{device.index()};

    Array actual_gx = EmptyLike(x, device);
    Array out_cont = AsContiguousArray(out);
    Array gout_cont = AsContiguousArray(gout);
    Array x_cont = AsContiguousArray(x);

    cuda_internal::CudnnTensorDescriptor out_desc{out_cont};
    cuda_internal::CudnnTensorDescriptor gout_desc{gout_cont};
    cuda_internal::CudnnTensorDescriptor x_desc{x_cont};
    cuda_internal::CudnnTensorDescriptor gx_desc{actual_gx};

    cuda_internal::CudnnPoolingDescriptor pool_desc{cudnn_pooling_mode, CUDNN_NOT_PROPAGATE_NAN, kernel_size, pad, stride};

    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);

    device_internals.cudnn_handle().Call(
            cudnnPoolingBackward,
            *pool_desc,
            cuda_internal::GetCudnnCoefficientPtr<1>(dtype),
            *out_desc,
            internal::GetRawOffsetData(out_cont),
            *gout_desc,
            internal::GetRawOffsetData(gout_cont),
            *x_desc,
            internal::GetRawOffsetData(x_cont),
            cuda_internal::GetCudnnCoefficientPtr<0>(dtype),
            *gx_desc,
            internal::GetRawOffsetData(actual_gx));

    return actual_gx;
}

Array MaxPoolGradGrad(
        const Array& x, const Array& out, const Array& ggx, Dims kernel_size, Dims stride, Dims pad, const absl::optional<Array>& ggout) {
    CHAINERX_ASSERT(x.shape() == ggx.shape());
    CHAINERX_ASSERT(kernel_size.size() == static_cast<size_t>(x.ndim() - 2));
    CHAINERX_ASSERT(stride.size() == static_cast<size_t>(x.ndim() - 2));
    CHAINERX_ASSERT(pad.size() == static_cast<size_t>(x.ndim() - 2));

    // TODO(hvy): Implement and test the `ggout` argument.
    if (ggout.has_value()) {
        throw NotImplementedError{"Passing ggout as an argument is not yet supported."};
    }

    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    if (ndim < 2) {
        throw DimensionError{"CUDA pooling requires number of spatial dimensions to be greater than or equal to 2"};
    }

    Device& device = ggx.device();
    CudaSetDeviceScope scope{device.index()};

    Array actual_ggout = EmptyLike(out, device);

    VisitFloatingPointDtype(actual_ggout.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        IndexableArray<const T> ggx_iarray{ggx};
        IndexableArray<const T> x_iarray{x};
        IndexableArray<const T> out_iarray{out};
        IndexableArray<T> ggout_iarray{actual_ggout};

        Indexer<> x_indexer{x.shape()};
        Indexer<> out_indexer{out.shape()};
        Indexer<> kernel_indexer{Shape{kernel_size.begin(), kernel_size.end()}};

        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&MaxPoolDoubleBackwardKernel<T>).block_size;
        int64_t total_size = out_indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        MaxPoolDoubleBackwardKernel<<<grid_size, block_size>>>(
                ggx_iarray, x_iarray, out_iarray, ggout_iarray, x_indexer, out_indexer, kernel_indexer, CudaDims{stride}, CudaDims{pad});
    });

    return actual_ggout;
}

class CudaMaxPoolKernel : public MaxPoolKernel {
public:
    std::tuple<Array, std::unique_ptr<MaxPoolGradState>> Call(
            const Array& x, Dims kernel_size, Dims stride, Dims pad, bool cover_all, bool return_state, const absl::optional<Array>& out)
            override {
        CHAINERX_ASSERT(internal::GetArrayBody(x)->nodes().empty());

        Array actual_out = Pool(CUDNN_POOLING_MAX, x, kernel_size, stride, pad, cover_all, out);

        std::unique_ptr<MaxPoolGradState> state = return_state ? std::make_unique<CudaMaxPoolGradState>(x, actual_out) : nullptr;

        return std::make_tuple(std::move(actual_out), std::move(state));
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(MaxPoolKernel, CudaMaxPoolKernel);

class CudaMaxPoolGradKernel : public MaxPoolGradKernel {
public:
    std::tuple<Array, std::unique_ptr<MaxPoolGradGradState>> Call(
            const Array& gout,
            const Dims& kernel_size,
            const Dims& stride,
            const Dims& pad,
            const std::shared_ptr<MaxPoolGradState>& state,
            bool return_state,
            const absl::optional<Array>& gx) override {
        CHAINERX_ASSERT(internal::GetArrayBody(gout)->nodes().empty());

        CHAINERX_ASSERT(state != nullptr);
        CudaMaxPoolGradState& cuda_state = dynamic_cast<CudaMaxPoolGradState&>(*state);
        const Array& x = cuda_state.x();
        const Array& out = cuda_state.out();

        Array actual_gx = PoolGrad(CUDNN_POOLING_MAX, x, out, gout, kernel_size, stride, pad, gx);

        std::unique_ptr<MaxPoolGradGradState> grad_grad_state = return_state ? std::make_unique<CudaMaxPoolGradGradState>(x, out) : nullptr;

        return std::make_tuple(std::move(actual_gx), std::move(grad_grad_state));
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(MaxPoolGradKernel, CudaMaxPoolGradKernel);

class CudaMaxPoolGradGradKernel : public MaxPoolGradGradKernel {
public:
    Array Call(
            const Array& ggx,
            const Dims& kernel_size,
            const Dims& stride,
            const Dims& pad,
            bool /*cover_all*/,
            const std::shared_ptr<MaxPoolGradGradState>& state,
            const absl::optional<Array>& ggout) override {
        CHAINERX_ASSERT(internal::GetArrayBody(ggx)->nodes().empty());

        CHAINERX_ASSERT(state != nullptr);
        CudaMaxPoolGradGradState& cuda_state = dynamic_cast<CudaMaxPoolGradGradState&>(*state);
        const Array& x = cuda_state.x();
        const Array& out = cuda_state.out();

        return MaxPoolGradGrad(x, out, ggx, kernel_size, stride, pad, ggout);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(MaxPoolGradGradKernel, CudaMaxPoolGradGradKernel);

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

class CudaAveragePoolKernel : public AveragePoolKernel {
public:
    std::tuple<Array, std::unique_ptr<AveragePoolGradState>> Call(
            const Array& x,
            const Dims& kernel_size,
            const Dims& stride,
            const Dims& pad,
            AveragePoolPadMode pad_mode,
            bool return_state,
            const absl::optional<Array>& out) override {
        CHAINERX_ASSERT(internal::GetArrayBody(x)->nodes().empty());

        Array actual_out = Pool(GetCudnnPoolingMode(pad_mode), x, kernel_size, stride, pad, false, out);

        std::unique_ptr<AveragePoolGradState> state = return_state ? std::make_unique<CudaAveragePoolGradState>(x, actual_out) : nullptr;

        return std::make_tuple(std::move(actual_out), std::move(state));
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(AveragePoolKernel, CudaAveragePoolKernel);

class CudaAveragePoolGradKernel : public AveragePoolGradKernel {
public:
    Array Call(
            const Array& gout,
            const Dims& kernel_size,
            const Dims& stride,
            const Dims& pad,
            AveragePoolPadMode pad_mode,
            const std::shared_ptr<AveragePoolGradState>& state,
            const absl::optional<Array>& gx) override {
        CHAINERX_ASSERT(internal::GetArrayBody(gout)->nodes().empty());

        CHAINERX_ASSERT(state != nullptr);
        CudaAveragePoolGradState& cuda_state = dynamic_cast<CudaAveragePoolGradState&>(*state);
        const Array& x = cuda_state.x();
        const Array& out = cuda_state.out();

        return PoolGrad(GetCudnnPoolingMode(pad_mode), x, out, gout, kernel_size, stride, pad, gx);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(AveragePoolGradKernel, CudaAveragePoolGradKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
