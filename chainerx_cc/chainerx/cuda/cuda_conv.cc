#include "chainerx/cuda/cuda_conv.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <utility>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cuda_backend.h"
#include "chainerx/cuda/cuda_device.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/cudnn.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/hash_combine.h"
#include "chainerx/macro.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {
namespace {

// Dtype configuration in Conv
struct ConvDtypes {
    Dtype in_dtype;
    Dtype conv_dtype;
    Dtype out_dtype;
};

ConvDtypes GetConvDtypes(Dtype in_dtype1, Dtype in_dtype2, Dtype out_dtype) {
    // TODO(imanishi): Support TRUE_HALF_CONFIG data type configuration if the compute capability is 5.3 or later.
    // TODO(niboshi): Devices with CC>=6.1 should support (int8, int32, int8) and (int8, int32, float32).

    if (in_dtype1 != in_dtype2) {
        in_dtype1 = out_dtype;
        in_dtype2 = out_dtype;
    }

    CHAINERX_ASSERT(in_dtype1 == in_dtype2);

    // Float output
    switch (in_dtype1) {
        case Dtype::kFloat16:  // fallthrough
        case Dtype::kFloat32:
            return ConvDtypes{in_dtype1, Dtype::kFloat32, out_dtype};
        case Dtype::kFloat64:
            return ConvDtypes{in_dtype2, Dtype::kFloat64, out_dtype};
        default:
            break;
    }

    CHAINERX_ASSERT(GetKind(in_dtype1) != DtypeKind::kFloat);

    // Non-float output
    return ConvDtypes{Dtype::kFloat64, Dtype::kFloat64, Dtype::kFloat64};
}

}  // namespace

namespace cuda_internal {

std::size_t CudaConv::AlgoCacheKeyHash::operator()(const AlgoCacheKey& key) const {
    std::size_t seed = 0;
    internal::HashCombine(seed, std::hash<int8_t>()(key.x_shape.ndim()));
    for (int64_t v : key.x_shape) {
        internal::HashCombine(seed, std::hash<int64_t>()(v));
    }
    internal::HashCombine(seed, std::hash<int8_t>()(key.w_shape.ndim()));
    for (int64_t v : key.w_shape) {
        internal::HashCombine(seed, std::hash<int64_t>()(v));
    }
    internal::HashCombine(seed, std::hash<int8_t>()(key.y_shape.ndim()));
    for (int64_t v : key.y_shape) {
        internal::HashCombine(seed, std::hash<int64_t>()(v));
    }
    internal::HashCombine(seed, std::hash<int8_t>()(gsl::narrow<int8_t>(key.pad.size())));
    for (int64_t v : key.pad) {
        internal::HashCombine(seed, std::hash<int64_t>()(v));
    }
    internal::HashCombine(seed, std::hash<int8_t>()(gsl::narrow<int8_t>(key.stride.size())));
    for (int64_t v : key.stride) {
        internal::HashCombine(seed, std::hash<int64_t>()(v));
    }
    internal::HashCombine(seed, std::hash<std::underlying_type_t<Dtype>>()(static_cast<std::underlying_type_t<Dtype>>(key.dtype)));
    internal::HashCombine(seed, std::hash<size_t>()(key.max_workspace_size));
    return seed;
}

void CudaConv::AddBias(CudnnHandle& handle, const CudnnTensorDescriptor& y_desc, const Array& y, const Array& b) {
    CHAINERX_ASSERT(&b.device() == &y.device());
    CHAINERX_ASSERT(b.dtype() == y.dtype());

    int8_t ndim = y.ndim() - 2;  // Number of spatial dimensions
    CHAINERX_ASSERT(ndim > 0);

    Shape new_shape{};
    new_shape.emplace_back(1);
    new_shape.emplace_back(b.GetTotalSize());
    for (int8_t i = 0; i < ndim; ++i) {
        new_shape.emplace_back(1);
    }
    Array b_cont = internal::AsContiguous(b).Reshape(new_shape);

    CudnnTensorDescriptor b_desc{b_cont};
    handle.Call(
            cudnnAddTensor,
            GetCudnnCoefficientPtr<1>(y.dtype()),
            *b_desc,
            internal::GetRawOffsetData(b_cont),
            GetCudnnCoefficientPtr<1>(y.dtype()),
            *y_desc,
            internal::GetRawOffsetData(y));
}

std::pair<cudnnConvolutionFwdAlgo_t, size_t> CudaConv::FindConvolutionForwardAlgorithm(
        CudnnHandle& handle,
        const CudnnTensorDescriptor& x_desc,
        const Array& x,
        const CudnnFilterDescriptor& filter_desc,
        const Array& w,
        const CudnnConvolutionDescriptor& conv_desc,
        const CudnnTensorDescriptor& y_desc,
        const Array& y,
        size_t max_workspace_size,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = AlgoCacheKey{x.shape(), w.shape(), y.shape(), pad, stride, x.dtype(), max_workspace_size};
    auto& algo_cache_map = fwd_algo_cache_map_;
    {
        std::lock_guard<std::mutex> lock{fwd_algo_cache_mutex_};
        auto it = algo_cache_map.find(key);
        if (it != algo_cache_map.end()) {
            return it->second;
        }
    }

    std::shared_ptr<void> workspace = y.device().Allocate(max_workspace_size);

    cudnnConvolutionFwdAlgoPerf_t perf_result{};
    int returned_algo_count{};

    handle.Call(
            cudnnFindConvolutionForwardAlgorithmEx,
            *x_desc,
            internal::GetRawOffsetData(x),
            *filter_desc,
            internal::GetRawOffsetData(w),
            *conv_desc,
            *y_desc,
            internal::GetRawOffsetData(y),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size);
    CHAINERX_ASSERT(returned_algo_count == 1);

    {
        std::lock_guard<std::mutex> lock{fwd_algo_cache_mutex_};
        return algo_cache_map[key] = {perf_result.algo, perf_result.memory};
    }
}

std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> CudaConv::FindConvolutionBackwardDataAlgorithm(
        CudnnHandle& handle,
        const CudnnFilterDescriptor& filter_desc,
        const Array& w,
        const CudnnTensorDescriptor& x_desc,
        const Array& x,
        const CudnnConvolutionDescriptor& conv_desc,
        const CudnnTensorDescriptor& y_desc,
        const Array& y,
        size_t max_workspace_size,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = AlgoCacheKey{x.shape(), w.shape(), y.shape(), pad, stride, x.dtype(), max_workspace_size};
    auto& algo_cache_map = bwd_data_algo_cache_map_;
    {
        std::lock_guard<std::mutex> lock{bwd_data_algo_cache_mutex_};
        auto it = algo_cache_map.find(key);
        if (it != algo_cache_map.end()) {
            return it->second;
        }
    }

    std::shared_ptr<void> workspace = y.device().Allocate(max_workspace_size);

    cudnnConvolutionBwdDataAlgoPerf_t perf_result{};
    int returned_algo_count{};

    handle.Call(
            cudnnFindConvolutionBackwardDataAlgorithmEx,
            *filter_desc,
            internal::GetRawOffsetData(w),
            *x_desc,
            internal::GetRawOffsetData(x),
            *conv_desc,
            *y_desc,
            internal::GetRawOffsetData(y),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size);
    CHAINERX_ASSERT(returned_algo_count == 1);

    {
        std::lock_guard<std::mutex> lock{bwd_data_algo_cache_mutex_};
        return algo_cache_map[key] = {perf_result.algo, perf_result.memory};
    }
}

std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t> CudaConv::FindConvolutionBackwardFilterAlgorithm(
        CudnnHandle& handle,
        const CudnnTensorDescriptor& x_desc,
        const Array& x,
        const CudnnTensorDescriptor& gy_desc,
        const Array& gy,
        const CudnnConvolutionDescriptor& conv_desc,
        const CudnnFilterDescriptor& gw_desc,
        const Array& gw,
        size_t max_workspace_size,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = AlgoCacheKey{x.shape(), gw.shape(), gy.shape(), pad, stride, x.dtype(), max_workspace_size};
    auto& algo_cache_map = bwd_filter_algo_cache_map_;
    {
        std::lock_guard<std::mutex> lock{bwd_filter_algo_cache_mutex_};
        auto it = algo_cache_map.find(key);
        if (it != algo_cache_map.end()) {
            return it->second;
        }
    }

    std::shared_ptr<void> workspace = x.device().Allocate(max_workspace_size);

    cudnnConvolutionBwdFilterAlgoPerf_t perf_result{};
    int returned_algo_count{};

    handle.Call(
            cudnnFindConvolutionBackwardFilterAlgorithmEx,
            *x_desc,
            internal::GetRawOffsetData(x),
            *gy_desc,
            internal::GetRawOffsetData(gy),
            *conv_desc,
            *gw_desc,
            internal::GetRawOffsetData(gw),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size);
    CHAINERX_ASSERT(returned_algo_count == 1);

    {
        std::lock_guard<std::mutex> lock{bwd_filter_algo_cache_mutex_};
        return algo_cache_map[key] = {perf_result.algo, perf_result.memory};
    }
}

// TODO(sonots): Support tensor core
Array CudaConv::Conv(
        CudaDevice& device,
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all,
        Dtype out_dtype) {
    if (cover_all) {
        throw ChainerxError{"CUDA convolution does not support cover_all"};
    }

    if (b) {
        device.CheckDevicesCompatible(x, w, *b);
    } else {
        device.CheckDevicesCompatible(x, w);
    }
    CudaSetDeviceScope scope{device.index()};

    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    if (ndim < 2) {
        throw DimensionError{"CUDA convolution requires number of spatial dimensions to be greater than or equal to 2"};
    }
    CHAINERX_ASSERT(w.ndim() == x.ndim());
    CHAINERX_ASSERT(stride.size() == static_cast<size_t>(ndim));
    CHAINERX_ASSERT(pad.size() == static_cast<size_t>(ndim));

    // w.shape = (out_channels, _, k_1, k_2, ..., k_N)
    int64_t out_channels = w.shape()[0];
    // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    int64_t batch_size = x.shape()[0];

    // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    Shape out_shape{batch_size, out_channels};
    for (int8_t i = 0; i < ndim; ++i) {
        out_shape.emplace_back(internal::GetConvOutDim(x.shape()[i + 2], w.shape()[i + 2], stride[i], pad[i], cover_all));
        CHAINERX_ASSERT(out_shape.back() > 0);
    }
    ConvDtypes dtypes = GetConvDtypes(x.dtype(), w.dtype(), out_dtype);

    Array y = Empty(out_shape, dtypes.out_dtype, device);

    auto& backend = static_cast<CudaBackend&>(device.backend());  // NOLINT

    const Array& x_cast = dtypes.in_dtype == x.dtype() ? x : x.AsType(dtypes.in_dtype);
    const Array& w_cast = dtypes.in_dtype == w.dtype() ? w : w.AsType(dtypes.in_dtype);

    Array x_cont = internal::AsContiguous(x_cast);
    Array w_cont = internal::AsContiguous(w_cast);

    CudnnTensorDescriptor x_desc{x_cont};
    CudnnTensorDescriptor y_desc{y};
    CudnnFilterDescriptor filter_desc{w_cont};
    CudnnConvolutionDescriptor conv_desc{dtypes.conv_dtype, pad, stride, nonstd::nullopt /*dilation*/, 1 /*groups*/};

    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();
    CudnnHandle& handle = device.cudnn_handle();

    // auto tune
    std::pair<cudnnConvolutionFwdAlgo_t, size_t> algo_workspace_size = FindConvolutionForwardAlgorithm(
            handle, x_desc, x_cont, filter_desc, w_cont, conv_desc, y_desc, y, max_workspace_size, pad, stride);

    cudnnConvolutionFwdAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    handle.Call(
            cudnnConvolutionForward,
            GetCudnnCoefficientPtr<1>(x_cont.dtype()),
            *x_desc,
            internal::GetRawOffsetData(x_cont),
            *filter_desc,
            internal::GetRawOffsetData(w_cont),
            *conv_desc,
            algo,
            workspace.get(),
            workspace_size,
            GetCudnnCoefficientPtr<0>(x_cont.dtype()),
            *y_desc,
            internal::GetRawOffsetData(y));

    if (b) {
        const Array& b_cast = b->dtype() == y.dtype() ? *b : b->AsType(y.dtype());
        AddBias(handle, y_desc, y, b_cast);
    }

    if (y.dtype() != out_dtype) {
        y = y.AsType(out_dtype);
    }

    return y;
}

Array CudaConv::ConvTranspose(
        CudaDevice& device,
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& out_size) {
    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions

    // Check if cover_all is false
    for (int8_t i = 0; i < ndim; ++i) {
        if (x.shape()[i + 2] != internal::GetConvOutDim(out_size[i], w.shape()[i + 2], stride[i], pad[i], false)) {
            throw ChainerxError{"CUDA transposed convolution does not support specified output sizes"};
        }
    }

    if (b) {
        device.CheckDevicesCompatible(x, w, *b);
    } else {
        device.CheckDevicesCompatible(x, w);
    }
    CudaSetDeviceScope scope{device.index()};

    if (ndim < 2) {
        throw DimensionError{"CUDA convolution requires number of spatial dimensions to be greater than or equal to 2"};
    }
    CHAINERX_ASSERT(w.ndim() == x.ndim());
    CHAINERX_ASSERT(stride.size() == static_cast<size_t>(ndim));
    CHAINERX_ASSERT(pad.size() == static_cast<size_t>(ndim));
    CHAINERX_ASSERT(out_size.size() == static_cast<size_t>(ndim));

    // w.shape = (in_channels, out_channels, k_1, k_2, ..., k_N)
    int64_t out_channels = w.shape()[1];
    // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    int64_t batch_size = x.shape()[0];

    // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    // (Note that cover_all is not supported in cuDNN implementation.)
    Shape out_shape{batch_size, out_channels};
    std::copy(out_size.begin(), out_size.end(), std::back_inserter(out_shape));

    Dtype out_dtype = x.dtype();  // TODO(niboshi): Specify as a parameter of this function
    ConvDtypes dtypes = GetConvDtypes(x.dtype(), w.dtype(), out_dtype);

    Array y = Empty(out_shape, dtypes.out_dtype, device);

    auto& backend = static_cast<CudaBackend&>(device.backend());  // NOLINT

    const Array& x_cast = dtypes.in_dtype == x.dtype() ? x : x.AsType(dtypes.in_dtype);
    const Array& w_cast = dtypes.in_dtype == w.dtype() ? w : w.AsType(dtypes.in_dtype);

    Array x_cont = internal::AsContiguous(x_cast);
    Array w_cont = internal::AsContiguous(w_cast);

    CudnnTensorDescriptor x_desc{x_cont};
    CudnnTensorDescriptor y_desc{y};
    CudnnFilterDescriptor filter_desc{w_cont};
    CudnnConvolutionDescriptor conv_desc{dtypes.conv_dtype, pad, stride, nonstd::nullopt /*dilation*/, 1 /*group*/};

    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();
    CudnnHandle& handle = device.cudnn_handle();

    // auto tune
    std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> algo_workspace_size = FindConvolutionBackwardDataAlgorithm(
            handle, filter_desc, w_cont, x_desc, x_cont, conv_desc, y_desc, y, max_workspace_size, pad, stride);

    cudnnConvolutionBwdDataAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    handle.Call(
            cudnnConvolutionBackwardData,
            GetCudnnCoefficientPtr<1>(x_cont.dtype()),
            *filter_desc,
            internal::GetRawOffsetData(w_cont),
            *x_desc,
            internal::GetRawOffsetData(x_cont),
            *conv_desc,
            algo,
            workspace.get(),
            workspace_size,
            GetCudnnCoefficientPtr<0>(x_cont.dtype()),
            *y_desc,
            internal::GetRawOffsetData(y));

    if (b) {
        const Array& b_cast = b->dtype() == y.dtype() ? *b : b->AsType(y.dtype());
        AddBias(handle, y_desc, y, b_cast);
    }

    if (y.dtype() != out_dtype) {
        y = y.AsType(out_dtype);
    }

    return y;
}

Array CudaConv::ConvGradWeight(
        CudaDevice& device,
        Dtype w_dtype,
        const Shape& w_shape,
        const Array& x,
        const Array& gy,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    if (cover_all) {
        throw ChainerxError{"CUDA convolution does not support cover_all"};
    }

    device.CheckDevicesCompatible(x, gy);
    CudaSetDeviceScope scope{device.index()};

    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    if (ndim < 2) {
        throw DimensionError{"CUDA convolution requires number of spatial dimensions to be greater than or equal to 2"};
    }

    CHAINERX_ASSERT(x.ndim() == w_shape.ndim());
    CHAINERX_ASSERT(stride.size() == static_cast<size_t>(ndim));
    CHAINERX_ASSERT(pad.size() == static_cast<size_t>(ndim));
    CHAINERX_ASSERT(gy.ndim() == w_shape.ndim());

    CHAINERX_ASSERT(x.dtype() == w_dtype);
    CHAINERX_ASSERT(x.dtype() == gy.dtype());

    if (CHAINERX_DEBUG) {
        // w_shape = (out_channels, in_channels, k_1, k_2, ..., k_N)
        int64_t out_channels = w_shape[0];
        // x.shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
        int64_t batch_size = x.shape()[0];
        // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
        Shape out_shape{batch_size, out_channels};
        for (int8_t i = 0; i < ndim; ++i) {
            out_shape.emplace_back(internal::GetConvOutDim(x.shape()[i + 2], w_shape[i + 2], stride[i], pad[i], cover_all));
            CHAINERX_ASSERT(out_shape.back() > 0);
        }
        CHAINERX_ASSERT(gy.shape() == out_shape);
    }

    ConvDtypes dtypes = GetConvDtypes(x.dtype(), gy.dtype(), w_dtype);
    Array gw = Empty(w_shape, dtypes.out_dtype, device);

    auto& backend = static_cast<CudaBackend&>(device.backend());  // NOLINT

    const Array& x_cast = dtypes.in_dtype == x.dtype() ? x : x.AsType(dtypes.in_dtype);
    const Array& gy_cast = dtypes.in_dtype == gy.dtype() ? gy : gy.AsType(dtypes.in_dtype);

    Array x_cont = internal::AsContiguous(x_cast);
    Array gy_cont = internal::AsContiguous(gy_cast);

    CudnnTensorDescriptor x_desc{x_cont};
    CudnnTensorDescriptor gy_desc{gy_cont};
    CudnnFilterDescriptor gw_desc{gw};
    CudnnConvolutionDescriptor conv_desc{dtypes.conv_dtype, pad, stride, nonstd::nullopt /*dilation*/, 1 /*groups*/};

    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();
    CudnnHandle& handle = device.cudnn_handle();

    // auto tune
    std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t> algo_workspace_size =
            FindConvolutionBackwardFilterAlgorithm(handle, x_desc, x, gy_desc, gy, conv_desc, gw_desc, gw, max_workspace_size, pad, stride);

    cudnnConvolutionBwdFilterAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    handle.Call(
            cudnnConvolutionBackwardFilter,
            GetCudnnCoefficientPtr<1>(x_cont.dtype()),
            *x_desc,
            internal::GetRawOffsetData(x_cont),
            *gy_desc,
            internal::GetRawOffsetData(gy_cont),
            *conv_desc,
            algo,
            workspace.get(),
            workspace_size,
            GetCudnnCoefficientPtr<0>(x_cont.dtype()),
            *gw_desc,
            internal::GetRawOffsetData(gw));

    if (gw.dtype() != w_dtype) {
        gw = gw.AsType(w_dtype);
    }

    return gw;
}

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
