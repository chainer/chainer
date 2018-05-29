#include "xchainer/cuda/cudnn.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <cudnn.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/backend_util.h"
#include "xchainer/cuda/cuda_device.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/cuda/hash_combine.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/routines/creation.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace cuda {

CudnnError::CudnnError(cudnnStatus_t status) : XchainerError{cudnnGetErrorString(status)}, status_{status} {}

void CheckCudnnError(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw CudnnError{status};
    }
}

namespace {

cudnnDataType_t GetCudnnDataType(Dtype dtype) {
    switch (dtype) {
        case Dtype::kFloat32:
            return CUDNN_DATA_FLOAT;
        case Dtype::kFloat64:
            return CUDNN_DATA_DOUBLE;
        // TODO(sonots): Support float16 if it becomes avaialable
        // case Dtype::kFloat16:
        //    return CUDNN_DATA_HALF;
        default:
            throw DtypeError{"Dtype ", dtype, " is not supported in cuDNN"};
    }
}

// Returns a pointer to a value of given type, allocated on the static storage.
template <int kValue>
const void* GetValuePtr(Dtype dtype) {
    static const float kFloat32Value = kValue;
    static const double kFloat64Value = kValue;

    switch (dtype) {
        case Dtype::kFloat64:
            return &kFloat64Value;
        case Dtype::kFloat32:
            return &kFloat32Value;
        default:
            XCHAINER_NEVER_REACH();
    }
}

template <typename T, typename U, typename... ErrorArgs>
T narrow(U u, const ErrorArgs&... error_args) {
    T t = static_cast<T>(u);
    if (static_cast<U>(t) != u) {
        throw XchainerError{error_args...};
    }
    return t;
}

StackVector<int, kMaxNdim> GetIntShape(const Shape& shape) {
    StackVector<int, kMaxNdim> int_shape;
    for (int8_t i = 0; i < shape.ndim(); ++i) {
        int_shape.emplace_back(narrow<int>(shape[i], "Casting the shape size: ", shape[i], " at dimension: ", i, " to int failed."));
    }
    return int_shape;
}

StackVector<int, kMaxNdim> GetIntStride(const StackVector<int64_t, kMaxNdim>& stride) {
    StackVector<int, kMaxNdim> int_stride;
    for (size_t i = 0; i < stride.size(); ++i) {
        int_stride.emplace_back(narrow<int>(stride[i], "Casting the stride: ", stride[i], " at dimension: ", i, " to int failed."));
    }
    return int_stride;
}

StackVector<int, kMaxNdim> GetIntPad(const StackVector<int64_t, kMaxNdim>& pad) {
    StackVector<int, kMaxNdim> int_pad;
    for (size_t i = 0; i < pad.size(); ++i) {
        int_pad.emplace_back(narrow<int>(pad[i], "Casting the pad: ", pad[i], " at dimension: ", i, " to int failed."));
    }
    return int_pad;
}

StackVector<int, kMaxNdim> GetIntDilation(const StackVector<int64_t, kMaxNdim>& dilation) {
    StackVector<int, kMaxNdim> int_dilation;
    for (size_t i = 0; i < dilation.size(); ++i) {
        int_dilation.emplace_back(narrow<int>(dilation[i], "Casting the dilation: ", dilation[i], " at dimension: ", i, " to int failed."));
    }
    return int_dilation;
}

// Returns strides divided by item_size
StackVector<int, kMaxNdim> GetIntArrayStrides(const Strides& strides, int64_t item_size) {
    StackVector<int, kMaxNdim> int_strides;
    for (int8_t i = 0; i < strides.ndim(); ++i) {
        int64_t v = strides[i] / item_size;
        int_strides.emplace_back(
                narrow<int>(v, "Casting the array stride: ", v, " (in number of items) at dimension: ", i, " to int failed."));
    }
    return int_strides;
}

void SetTensorDescriptor(cudnnTensorDescriptor_t desc, const Array& arr, cudnnTensorFormat_t format) {
    assert(arr.IsContiguous());
    cudnnDataType_t cudnn_dtype = GetCudnnDataType(arr.dtype());
    if (arr.shape().ndim() == 4) {
        StackVector<int, kMaxNdim> nchw = GetIntShape(arr.shape());
        CheckCudnnError(cudnnSetTensor4dDescriptor(desc, format, cudnn_dtype, nchw[0], nchw[1], nchw[2], nchw[3]));
    } else {
        StackVector<int, kMaxNdim> int_strides = GetIntArrayStrides(arr.strides(), arr.item_size());  // strides divided by item_size
        StackVector<int, kMaxNdim> int_shape = GetIntShape(arr.shape());
        CheckCudnnError(cudnnSetTensorNdDescriptor(desc, cudnn_dtype, arr.ndim(), &int_shape[0], &int_strides[0]));
    }
}

void SetFilterDescriptor(cudnnFilterDescriptor_t desc, const Array& arr, cudnnTensorFormat_t format) {
    assert(arr.IsContiguous());
    cudnnDataType_t cudnn_dtype = GetCudnnDataType(arr.dtype());
    if (arr.shape().ndim() == 4) {
        StackVector<int, kMaxNdim> nchw = GetIntShape(arr.shape());
        CheckCudnnError(cudnnSetFilter4dDescriptor(desc, cudnn_dtype, format, nchw[0], nchw[1], nchw[2], nchw[3]));
    } else {
        StackVector<int, kMaxNdim> int_shape = GetIntShape(arr.shape());
        CheckCudnnError(cudnnSetFilterNdDescriptor(desc, cudnn_dtype, format, arr.ndim(), &int_shape[0]));
    }
}

void SetConvolutionDescriptor(
        cudnnConvolutionDescriptor_t desc,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups,
        Dtype dtype,
        cudnnConvolutionMode_t mode) {
    size_t ndim = pad.size();
    assert(ndim == stride.size());
    assert(!dilation || ndim == dilation->size());

    StackVector<int, kMaxNdim> int_stride = GetIntStride(stride);
    StackVector<int, kMaxNdim> int_pad = GetIntPad(pad);
    StackVector<int, kMaxNdim> int_dilation{};
    if (!dilation) {
        // TODO(sonots): Use assign(ndim, 1) if it becomes available
        for (size_t i = 0; i < ndim; ++i) {
            int_dilation.emplace_back(1);
        }
    } else {
        int_dilation = GetIntDilation(*dilation);
    }

    cudnnDataType_t compute_type = GetCudnnDataType(dtype);

    if (ndim == 2) {
        CheckCudnnError(cudnnSetConvolution2dDescriptor(
                desc, int_pad[0], int_pad[1], int_stride[0], int_stride[1], int_dilation[0], int_dilation[1], mode, compute_type));
    } else {
        CheckCudnnError(cudnnSetConvolutionNdDescriptor(desc, ndim, &int_pad[0], &int_stride[0], &int_dilation[0], mode, compute_type));
    }
    if (groups > 1) {
        CheckCudnnError(cudnnSetConvolutionGroupCount(desc, groups));
    }
}

std::shared_ptr<cudnnTensorStruct> CreateTensorDescriptor(const Array& arr, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW) {
    cudnnTensorDescriptor_t desc{};
    CheckCudnnError(cudnnCreateTensorDescriptor(&desc));
    auto shared_desc = std::shared_ptr<cudnnTensorStruct>{
            desc, [](cudnnTensorDescriptor_t desc) { CheckCudnnError(cudnnDestroyTensorDescriptor(desc)); }};
    SetTensorDescriptor(desc, arr, format);
    return shared_desc;
}

std::shared_ptr<cudnnFilterStruct> CreateFilterDescriptor(const Array& arr, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW) {
    cudnnFilterDescriptor_t desc{};
    CheckCudnnError(cudnnCreateFilterDescriptor(&desc));
    auto shared_desc = std::shared_ptr<cudnnFilterStruct>{
            desc, [](cudnnFilterDescriptor_t desc) { CheckCudnnError(cudnnDestroyFilterDescriptor(desc)); }};
    SetFilterDescriptor(desc, arr, format);
    return shared_desc;
}

std::shared_ptr<cudnnConvolutionStruct> CreateConvolutionDescriptor(
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        Dtype dtype,
        cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation = nonstd::nullopt,
        int groups = 1) {
    cudnnConvolutionDescriptor_t desc{};
    CheckCudnnError(cudnnCreateConvolutionDescriptor(&desc));
    auto shared_desc = std::shared_ptr<cudnnConvolutionStruct>{
            desc, [](cudnnConvolutionDescriptor_t desc) { CheckCudnnError(cudnnDestroyConvolutionDescriptor(desc)); }};
    SetConvolutionDescriptor(desc, pad, stride, dilation, groups, dtype, mode);
    return shared_desc;
}

}  // namespace

namespace internal {

void CudnnContext::AddBias(const std::shared_ptr<cudnnTensorStruct>& y_desc, const Array& y, const Array& b) {
    assert(&b.device() == &y.device());
    assert(b.dtype() == y.dtype());

    int8_t ndim = y.ndim() - 2;  // Number of spacial dimensions
    assert(ndim > 0);

    Shape new_shape{};
    new_shape.emplace_back(1);
    new_shape.emplace_back(b.GetTotalSize());
    for (int8_t i = 0; i < ndim; ++i) {
        new_shape.emplace_back(1);
    }
    Array b_cont = AsContiguousArray(b).Reshape(new_shape);

    std::shared_ptr<cudnnTensorStruct> b_desc = CreateTensorDescriptor(b_cont);
    CheckCudnnError(cudnnAddTensor(
            handle(),
            GetValuePtr<1>(y.dtype()),
            b_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(b_cont),
            GetValuePtr<1>(y.dtype()),
            y_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(y)));
}

std::size_t ConvAlgoCacheKeyHash::operator()(const ConvAlgoCacheKey& key) const {
    std::size_t seed = 0;
    HashCombine(seed, std::hash<int8_t>()(key.x_shape.ndim()));
    for (int64_t v : key.x_shape) {
        HashCombine(seed, std::hash<int64_t>()(v));
    }
    HashCombine(seed, std::hash<int8_t>()(key.w_shape.ndim()));
    for (int64_t v : key.w_shape) {
        HashCombine(seed, std::hash<int64_t>()(v));
    }
    HashCombine(seed, std::hash<int8_t>()(key.y_shape.ndim()));
    for (int64_t v : key.y_shape) {
        HashCombine(seed, std::hash<int64_t>()(v));
    }
    HashCombine(seed, std::hash<int8_t>()(gsl::narrow<int8_t>(key.pad.size())));
    for (int64_t v : key.pad) {
        HashCombine(seed, std::hash<int64_t>()(v));
    }
    HashCombine(seed, std::hash<int8_t>()(gsl::narrow<int8_t>(key.stride.size())));
    for (int64_t v : key.stride) {
        HashCombine(seed, std::hash<int64_t>()(v));
    }
    HashCombine(seed, std::hash<std::underlying_type_t<Dtype>>()(static_cast<std::underlying_type_t<Dtype>>(key.dtype)));
    HashCombine(seed, std::hash<size_t>()(key.max_workspace_size));
    return seed;
}

CudnnContext::CudnnContext(int device_index) : device_index_{device_index} {
    CheckCudaError(cudaSetDevice(device_index_));
    CheckCudnnError(cudnnCreate(&handle_));
}

CudnnContext::~CudnnContext() {
    cudaSetDevice(device_index_);
    cudnnDestroy(handle_);
}

std::pair<cudnnConvolutionFwdAlgo_t, size_t> CudnnContext::FindConvolutionForwardAlgorithm(
        const std::shared_ptr<cudnnTensorStruct>& x_desc,
        const Array& x,
        const std::shared_ptr<cudnnFilterStruct>& filter_desc,
        const Array& w,
        const std::shared_ptr<cudnnConvolutionStruct>& conv_desc,
        const std::shared_ptr<cudnnTensorStruct>& y_desc,
        const Array& y,
        size_t max_workspace_size,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = internal::ConvAlgoCacheKey{x.shape(), w.shape(), y.shape(), pad, stride, x.dtype(), max_workspace_size};
    if (conv_fwd_algo_cache_map_.count(key)) {
        return conv_fwd_algo_cache_map_[key];
    }

    std::shared_ptr<void> workspace = y.device().Allocate(max_workspace_size);

    cudnnConvolutionFwdAlgoPerf_t perf_result{};
    int returned_algo_count{};

    CheckCudnnError(cudnnFindConvolutionForwardAlgorithmEx(
            handle(),
            x_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(x),
            filter_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(w),
            conv_desc.get(),
            y_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(y),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size));
    assert(returned_algo_count == 1);

    return conv_fwd_algo_cache_map_[key] = {perf_result.algo, perf_result.memory};
}

std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> CudnnContext::FindConvolutionBackwardDataAlgorithm(
        const std::shared_ptr<cudnnFilterStruct>& filter_desc,
        const Array& w,
        const std::shared_ptr<cudnnTensorStruct>& x_desc,
        const Array& x,
        const std::shared_ptr<cudnnConvolutionStruct>& conv_desc,
        const std::shared_ptr<cudnnTensorStruct>& y_desc,
        const Array& y,
        size_t max_workspace_size,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = internal::ConvAlgoCacheKey{x.shape(), w.shape(), y.shape(), pad, stride, x.dtype(), max_workspace_size};
    if (conv_bwd_data_algo_cache_map_.count(key)) {
        return conv_bwd_data_algo_cache_map_[key];
    }

    std::shared_ptr<void> workspace = y.device().Allocate(max_workspace_size);

    cudnnConvolutionBwdDataAlgoPerf_t perf_result{};
    int returned_algo_count{};

    CheckCudnnError(cudnnFindConvolutionBackwardDataAlgorithmEx(
            handle(),
            filter_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(w),
            x_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(x),
            conv_desc.get(),
            y_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(y),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size));
    assert(returned_algo_count == 1);

    return conv_bwd_data_algo_cache_map_[key] = {perf_result.algo, perf_result.memory};
}

std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t> CudnnContext::FindConvolutionBackwardFilterAlgorithm(
        const std::shared_ptr<cudnnTensorStruct>& x_desc,
        const Array& x,
        const std::shared_ptr<cudnnTensorStruct>& gy_desc,
        const Array& gy,
        const std::shared_ptr<cudnnConvolutionStruct>& conv_desc,
        const std::shared_ptr<cudnnFilterStruct>& gw_desc,
        const Array& gw,
        size_t max_workspace_size,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = internal::ConvAlgoCacheKey{x.shape(), gw.shape(), gy.shape(), pad, stride, x.dtype(), max_workspace_size};
    if (conv_bwd_filter_algo_cache_map_.count(key)) {
        return conv_bwd_filter_algo_cache_map_[key];
    }

    std::shared_ptr<void> workspace = x.device().Allocate(max_workspace_size);

    cudnnConvolutionBwdFilterAlgoPerf_t perf_result{};
    int returned_algo_count{};

    CheckCudnnError(cudnnFindConvolutionBackwardFilterAlgorithmEx(
            handle(),
            x_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(x),
            gy_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(gy),
            conv_desc.get(),
            gw_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(gw),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size));
    assert(returned_algo_count == 1);

    return conv_bwd_filter_algo_cache_map_[key] = {perf_result.algo, perf_result.memory};
}

// TODO(sonots): Support tensor core
void CudnnContext::ConvolutionForward(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const Array& y,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups) {
    assert(&y.device() == &x.device());
    assert(y.dtype() == x.dtype());
    assert(&w.device() == &x.device());
    assert(w.dtype() == x.dtype());

    CudaDevice& device = static_cast<CudaDevice&>(x.device());
    CudaBackend& backend = static_cast<CudaBackend&>(device.backend());

    Array x_cont = AsContiguousArray(x);
    Array w_cont = AsContiguousArray(w);
    assert(y.IsContiguous());

    std::shared_ptr<cudnnTensorStruct> x_desc = CreateTensorDescriptor(x_cont);
    std::shared_ptr<cudnnTensorStruct> y_desc = CreateTensorDescriptor(y);
    std::shared_ptr<cudnnFilterStruct> filter_desc = CreateFilterDescriptor(w_cont, CUDNN_TENSOR_NCHW);
    std::shared_ptr<cudnnConvolutionStruct> conv_desc =
            CreateConvolutionDescriptor(pad, stride, x.dtype(), CUDNN_CROSS_CORRELATION, dilation, groups);
    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();

    // auto tune
    std::pair<cudnnConvolutionFwdAlgo_t, size_t> algo_workspace_size =
            FindConvolutionForwardAlgorithm(x_desc, x_cont, filter_desc, w_cont, conv_desc, y_desc, y, max_workspace_size, pad, stride);

    cudnnConvolutionFwdAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    CheckCudnnError(cudnnConvolutionForward(
            handle(),
            GetValuePtr<1>(x.dtype()),
            x_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(x_cont),
            filter_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(w_cont),
            conv_desc.get(),
            algo,
            workspace.get(),
            workspace_size,
            GetValuePtr<0>(x.dtype()),
            y_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(y)));

    if (b) {
        AddBias(y_desc, y, *b);
    }
}

void CudnnContext::ConvolutionBackwardData(
        const Array& w,
        const Array& x,
        const nonstd::optional<Array>& b,
        const Array& y,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups) {
    assert(&y.device() == &x.device());
    assert(y.dtype() == x.dtype());
    assert(&w.device() == &x.device());
    assert(w.dtype() == x.dtype());

    CudaDevice& device = static_cast<CudaDevice&>(x.device());
    CudaBackend& backend = static_cast<CudaBackend&>(device.backend());

    Array x_cont = AsContiguousArray(x);
    Array w_cont = AsContiguousArray(w);
    assert(y.IsContiguous());

    std::shared_ptr<cudnnTensorStruct> x_desc = CreateTensorDescriptor(x_cont);
    std::shared_ptr<cudnnTensorStruct> y_desc = CreateTensorDescriptor(y);
    std::shared_ptr<cudnnFilterStruct> filter_desc = CreateFilterDescriptor(w_cont, CUDNN_TENSOR_NCHW);
    std::shared_ptr<cudnnConvolutionStruct> conv_desc =
            CreateConvolutionDescriptor(pad, stride, x.dtype(), CUDNN_CROSS_CORRELATION, dilation, groups);
    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();

    // auto tune
    std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> algo_workspace_size = FindConvolutionBackwardDataAlgorithm(
            filter_desc, w_cont, x_desc, x_cont, conv_desc, y_desc, y, max_workspace_size, pad, stride);

    cudnnConvolutionBwdDataAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    CheckCudnnError(cudnnConvolutionBackwardData(
            handle(),
            GetValuePtr<1>(x.dtype()),
            filter_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(w_cont),
            x_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(x_cont),
            conv_desc.get(),
            algo,
            workspace.get(),
            workspace_size,
            GetValuePtr<0>(x.dtype()),
            y_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(y)));

    if (b) {
        AddBias(y_desc, y, *b);
    }
}

void CudnnContext::ConvolutionBackwardFilter(
        const Array& x,
        const Array& gy,
        const Array& gw,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups) {
    assert(&x.device() == &gy.device());
    assert(&x.device() == &gw.device());
    assert(x.dtype() == gy.dtype());
    assert(x.dtype() == gw.dtype());

    CudaDevice& device = static_cast<CudaDevice&>(x.device());
    CudaBackend& backend = static_cast<CudaBackend&>(device.backend());

    Array x_cont = AsContiguousArray(x);
    Array gy_cont = AsContiguousArray(gy);
    Array gw_cont = AsContiguousArray(gw);

    std::shared_ptr<cudnnTensorStruct> x_desc = CreateTensorDescriptor(x_cont);
    std::shared_ptr<cudnnTensorStruct> gy_desc = CreateTensorDescriptor(gy_cont);
    std::shared_ptr<cudnnFilterStruct> gw_desc = CreateFilterDescriptor(gw_cont, CUDNN_TENSOR_NCHW);
    std::shared_ptr<cudnnConvolutionStruct> conv_desc =
            CreateConvolutionDescriptor(pad, stride, x.dtype(), CUDNN_CROSS_CORRELATION, dilation, groups);
    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();

    // auto tune
    std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t> algo_workspace_size =
            FindConvolutionBackwardFilterAlgorithm(x_desc, x, gy_desc, gy, conv_desc, gw_desc, gw, max_workspace_size, pad, stride);

    cudnnConvolutionBwdFilterAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    CheckCudnnError(cudnnConvolutionBackwardFilter(
            handle(),
            GetValuePtr<1>(x.dtype()),
            x_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(x_cont),
            gy_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(gy_cont),
            conv_desc.get(),
            algo,
            workspace.get(),
            workspace_size,
            GetValuePtr<0>(x.dtype()),
            gw_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(gw)));
}

}  // namespace internal
}  // namespace cuda
}  // namespace xchainer
