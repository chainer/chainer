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

}  // namespace

namespace internal {

void CudnnContext::AddBias(const TensorDescriptor& y_desc, const Array& y, const Array& b) {
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

    TensorDescriptor b_desc{b_cont};
    CheckCudnnError(cudnnAddTensor(
            handle(),
            GetValuePtr<1>(y.dtype()),
            *b_desc,
            xchainer::internal::GetRawOffsetData<void>(b_cont),
            GetValuePtr<1>(y.dtype()),
            *y_desc,
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

void CudnnContext::BatchNormalizationForwardTraining(
        BatchNormMode mode,
        const Array& x,
        const Array& y,
        const Array& scale,  // gamma
        const Array& bias,  // beta
        double exponential_average_factor,
        const Array& result_running_mean,
        const Array& result_running_variance,
        double eps,
        const nonstd::optional<Array>& result_save_mean,
        const nonstd::optional<Array>& result_save_inv_variance) {
    if (eps < CUDNN_BN_MIN_EPSILON) {
        throw CudnnError{"Minimum allowed epsilon is ", CUDNN_BN_MIN_EPSILON, " but found ", eps, "."};
    }

    Device& device = x.device();
    Dtype dtype = x.dtype();
    assert(&device == &y.device());
    assert(&device == &scale.device());
    assert(&device == &bias.device());
    assert(&device == &result_running_mean.device());
    assert(&device == &result_running_variance.device());
    assert(result_save_mean.has_value() ==
           result_save_inv_variance.has_value());  // Caches can be omitted but only at the same time for the mean and inverse variance.

    bool cache_mean_and_inv_variance = result_save_mean.has_value();

    assert(dtype == y.dtype());
    assert(dtype == scale.dtype());
    assert(dtype == bias.dtype());
    assert(dtype == result_running_mean.dtype());
    assert(dtype == result_running_variance.dtype());
    if (cache_mean_and_inv_variance) {
        assert(dtype == result_save_mean->dtype());
        assert(dtype == result_save_inv_variance->dtype());
    }

    Array x_cont = AsContiguousArray(x);
    Array scale_cont = AsContiguousArray(scale);
    Array bias_cont = AsContiguousArray(bias);
    Array result_running_mean_cont = AsContiguousArray(result_running_mean);
    Array result_running_variance_cont = AsContiguousArray(result_running_variance);
    assert(y.IsContiguous());

    TensorDescriptor x_desc{x_cont};
    TensorDescriptor y_desc{y};
    TensorDescriptor scale_bias_mean_var_desc{result_running_mean_cont};

    void* result_save_mean_raw;
    void* result_save_inv_variance_raw;
    if (cache_mean_and_inv_variance) {
        assert(result_save_mean->IsContiguous());
        assert(result_save_inv_variance->IsContiguous());
        result_save_mean_raw = xchainer::internal::GetRawOffsetData<void>(*result_save_mean);
        result_save_inv_variance_raw = xchainer::internal::GetRawOffsetData<void>(*result_save_inv_variance);
    } else {
        result_save_mean_raw = nullptr;
        result_save_inv_variance_raw = nullptr;
    }

    CheckCudnnError(cudnnBatchNormalizationForwardTraining(
            handle(),
            mode,
            GetValuePtr<1>(x.dtype()),
            GetValuePtr<0>(x.dtype()),
            *x_desc,
            xchainer::internal::GetRawOffsetData<void>(x_cont),
            *y_desc,
            xchainer::internal::GetRawOffsetData<void>(y),
            *scale_bias_mean_var_desc,
            xchainer::internal::GetRawOffsetData<void>(scale_cont),
            xchainer::internal::GetRawOffsetData<void>(bias_cont),
            exponential_average_factor,
            xchainer::internal::GetRawOffsetData<void>(result_running_mean_cont),
            xchainer::internal::GetRawOffsetData<void>(result_running_variance_cont),
            eps,
            result_save_mean_raw,
            result_save_inv_variance_raw));
}

std::pair<cudnnConvolutionFwdAlgo_t, size_t> CudnnContext::FindConvolutionForwardAlgorithm(
        const TensorDescriptor& x_desc,
        const Array& x,
        const FilterDescriptor& filter_desc,
        const Array& w,
        const ConvolutionDescriptor& conv_desc,
        const TensorDescriptor& y_desc,
        const Array& y,
        size_t max_workspace_size,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = internal::ConvAlgoCacheKey{x.shape(), w.shape(), y.shape(), pad, stride, x.dtype(), max_workspace_size};
    auto& algo_cache_map = conv_fwd_algo_cache_map_;
    auto it = algo_cache_map.find(key);
    if (it != algo_cache_map.end()) {
        return it->second;
    }

    std::shared_ptr<void> workspace = y.device().Allocate(max_workspace_size);

    cudnnConvolutionFwdAlgoPerf_t perf_result{};
    int returned_algo_count{};

    CheckCudnnError(cudnnFindConvolutionForwardAlgorithmEx(
            handle(),
            *x_desc,
            xchainer::internal::GetRawOffsetData<void>(x),
            *filter_desc,
            xchainer::internal::GetRawOffsetData<void>(w),
            *conv_desc,
            *y_desc,
            xchainer::internal::GetRawOffsetData<void>(y),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size));
    assert(returned_algo_count == 1);

    return algo_cache_map[key] = {perf_result.algo, perf_result.memory};
}

std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> CudnnContext::FindConvolutionBackwardDataAlgorithm(
        const FilterDescriptor& filter_desc,
        const Array& w,
        const TensorDescriptor& x_desc,
        const Array& x,
        const ConvolutionDescriptor& conv_desc,
        const TensorDescriptor& y_desc,
        const Array& y,
        size_t max_workspace_size,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = internal::ConvAlgoCacheKey{x.shape(), w.shape(), y.shape(), pad, stride, x.dtype(), max_workspace_size};
    auto& algo_cache_map = conv_bwd_data_algo_cache_map_;
    auto it = algo_cache_map.find(key);
    if (it != algo_cache_map.end()) {
        return it->second;
    }

    std::shared_ptr<void> workspace = y.device().Allocate(max_workspace_size);

    cudnnConvolutionBwdDataAlgoPerf_t perf_result{};
    int returned_algo_count{};

    CheckCudnnError(cudnnFindConvolutionBackwardDataAlgorithmEx(
            handle(),
            *filter_desc,
            xchainer::internal::GetRawOffsetData<void>(w),
            *x_desc,
            xchainer::internal::GetRawOffsetData<void>(x),
            *conv_desc,
            *y_desc,
            xchainer::internal::GetRawOffsetData<void>(y),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size));
    assert(returned_algo_count == 1);

    return algo_cache_map[key] = {perf_result.algo, perf_result.memory};
}

std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t> CudnnContext::FindConvolutionBackwardFilterAlgorithm(
        const TensorDescriptor& x_desc,
        const Array& x,
        const TensorDescriptor& gy_desc,
        const Array& gy,
        const ConvolutionDescriptor& conv_desc,
        const FilterDescriptor& gw_desc,
        const Array& gw,
        size_t max_workspace_size,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = internal::ConvAlgoCacheKey{x.shape(), gw.shape(), gy.shape(), pad, stride, x.dtype(), max_workspace_size};
    auto& algo_cache_map = conv_bwd_filter_algo_cache_map_;
    auto it = algo_cache_map.find(key);
    if (it != algo_cache_map.end()) {
        return it->second;
    }

    std::shared_ptr<void> workspace = x.device().Allocate(max_workspace_size);

    cudnnConvolutionBwdFilterAlgoPerf_t perf_result{};
    int returned_algo_count{};

    CheckCudnnError(cudnnFindConvolutionBackwardFilterAlgorithmEx(
            handle(),
            *x_desc,
            xchainer::internal::GetRawOffsetData<void>(x),
            *gy_desc,
            xchainer::internal::GetRawOffsetData<void>(gy),
            *conv_desc,
            *gw_desc,
            xchainer::internal::GetRawOffsetData<void>(gw),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size));
    assert(returned_algo_count == 1);

    return algo_cache_map[key] = {perf_result.algo, perf_result.memory};
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

    auto& device = static_cast<CudaDevice&>(x.device());
    auto& backend = static_cast<CudaBackend&>(device.backend());

    Array x_cont = AsContiguousArray(x);
    Array w_cont = AsContiguousArray(w);
    assert(y.IsContiguous());

    TensorDescriptor x_desc{x_cont};
    TensorDescriptor y_desc{y};
    FilterDescriptor filter_desc{w_cont};
    ConvolutionDescriptor conv_desc{x.dtype(), pad, stride, dilation, groups};
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
            *x_desc,
            xchainer::internal::GetRawOffsetData<void>(x_cont),
            *filter_desc,
            xchainer::internal::GetRawOffsetData<void>(w_cont),
            *conv_desc,
            algo,
            workspace.get(),
            workspace_size,
            GetValuePtr<0>(x.dtype()),
            *y_desc,
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

    auto& device = static_cast<CudaDevice&>(x.device());
    auto& backend = static_cast<CudaBackend&>(device.backend());

    Array x_cont = AsContiguousArray(x);
    Array w_cont = AsContiguousArray(w);
    assert(y.IsContiguous());

    TensorDescriptor x_desc{x_cont};
    TensorDescriptor y_desc{y};
    FilterDescriptor filter_desc{w_cont};
    ConvolutionDescriptor conv_desc{x.dtype(), pad, stride, dilation, groups};
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
            *filter_desc,
            xchainer::internal::GetRawOffsetData<void>(w_cont),
            *x_desc,
            xchainer::internal::GetRawOffsetData<void>(x_cont),
            *conv_desc,
            algo,
            workspace.get(),
            workspace_size,
            GetValuePtr<0>(x.dtype()),
            *y_desc,
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

    TensorDescriptor x_desc{x_cont};
    TensorDescriptor gy_desc{gy_cont};
    FilterDescriptor gw_desc{gw_cont};
    ConvolutionDescriptor conv_desc{x.dtype(), pad, stride, dilation, groups};
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
            *x_desc,
            xchainer::internal::GetRawOffsetData<void>(x_cont),
            *gy_desc,
            xchainer::internal::GetRawOffsetData<void>(gy_cont),
            *conv_desc,
            algo,
            workspace.get(),
            workspace_size,
            GetValuePtr<0>(x.dtype()),
            *gw_desc,
            xchainer::internal::GetRawOffsetData<void>(gw)));
}

CudnnContext::ConvolutionDescriptor::ConvolutionDescriptor() { CheckCudnnError(cudnnCreateConvolutionDescriptor(&desc_)); }

CudnnContext::ConvolutionDescriptor::~ConvolutionDescriptor() {
    if (desc_ != nullptr) {
        CheckCudnnError(cudnnDestroyConvolutionDescriptor(desc_));
    }
}

CudnnContext::ConvolutionDescriptor::ConvolutionDescriptor(
        Dtype dtype,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups)
    : ConvolutionDescriptor{} {
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
                desc_,
                int_pad[0],
                int_pad[1],
                int_stride[0],
                int_stride[1],
                int_dilation[0],
                int_dilation[1],
                CUDNN_CROSS_CORRELATION,
                compute_type));
    } else {
        CheckCudnnError(cudnnSetConvolutionNdDescriptor(
                desc_, ndim, &int_pad[0], &int_stride[0], &int_dilation[0], CUDNN_CROSS_CORRELATION, compute_type));
    }
    if (groups > 1) {
        CheckCudnnError(cudnnSetConvolutionGroupCount(desc_, groups));
    }
}

CudnnContext::TensorDescriptor::TensorDescriptor() { CheckCudnnError(cudnnCreateTensorDescriptor(&desc_)); }

CudnnContext::TensorDescriptor::~TensorDescriptor() {
    if (desc_ != nullptr) {
        CheckCudnnError(cudnnDestroyTensorDescriptor(desc_));
    }
}

CudnnContext::TensorDescriptor::TensorDescriptor(const Array& arr) : TensorDescriptor{} {
    assert(arr.IsContiguous());

    cudnnDataType_t cudnn_dtype = GetCudnnDataType(arr.dtype());
    if (arr.shape().ndim() == 4) {
        StackVector<int, kMaxNdim> nchw = GetIntShape(arr.shape());
        CheckCudnnError(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, cudnn_dtype, nchw[0], nchw[1], nchw[2], nchw[3]));
    } else {
        StackVector<int, kMaxNdim> int_strides = GetIntArrayStrides(arr.strides(), arr.item_size());  // strides divided by item_size
        StackVector<int, kMaxNdim> int_shape = GetIntShape(arr.shape());
        CheckCudnnError(cudnnSetTensorNdDescriptor(desc_, cudnn_dtype, arr.ndim(), &int_shape[0], &int_strides[0]));
    }
}

CudnnContext::FilterDescriptor::FilterDescriptor() { CheckCudnnError(cudnnCreateFilterDescriptor(&desc_)); }

CudnnContext::FilterDescriptor::~FilterDescriptor() {
    if (desc_ != nullptr) {
        CheckCudnnError(cudnnDestroyFilterDescriptor(desc_));
    }
}

CudnnContext::FilterDescriptor::FilterDescriptor(const Array& w) : FilterDescriptor{} {
    assert(w.IsContiguous());

    cudnnDataType_t cudnn_dtype = GetCudnnDataType(w.dtype());
    if (w.shape().ndim() == 4) {
        StackVector<int, kMaxNdim> nchw = GetIntShape(w.shape());
        CheckCudnnError(cudnnSetFilter4dDescriptor(desc_, cudnn_dtype, CUDNN_TENSOR_NCHW, nchw[0], nchw[1], nchw[2], nchw[3]));
    } else {
        StackVector<int, kMaxNdim> int_shape = GetIntShape(w.shape());
        CheckCudnnError(cudnnSetFilterNdDescriptor(desc_, cudnn_dtype, CUDNN_TENSOR_NCHW, w.ndim(), &int_shape[0]));
    }
}

}  // namespace internal
}  // namespace cuda
}  // namespace xchainer
