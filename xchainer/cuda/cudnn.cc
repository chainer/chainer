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

    std::pair<cudnnConvolutionFwdAlgo_t, size_t> algo_memory = {perf_result.algo, perf_result.memory};
    conv_fwd_algo_cache_map_[key] = algo_memory;
    return algo_memory;
}


#if 0
cpdef tuple _find_algorithm_bwd_data(
        core.ndarray W, core.ndarray x, core.ndarray y, tuple conv_param,
        size_t handle, size_t filter_desc, size_t x_desc, size_t conv_desc,
        size_t y_desc, size_t max_workspace_size):
    key = (x.data.device.id, W.shape, x.shape, y.shape, conv_param,
           max_workspace_size)
    if key in _algorithm_bwd_data:
        return _algorithm_bwd_data[key]
    workspace = memory.alloc(max_workspace_size)
    ret = cudnn.findConvolutionBackwardDataAlgorithmEx(
        handle, filter_desc, W.data.ptr, x_desc, x.data.ptr,
        conv_desc, y_desc, y.data.ptr, 1, workspace.ptr, max_workspace_size)
    algo = (ret[0]['algo'], ret[0]['memory'])
    _algorithm_bwd_data[key] = algo
    return algo
#endif

std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> Cudnn::FindConvolutionBackwardDataAlgorithm(
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

    int requested_algo_count = 1;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results[1]{};
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
            requested_algo_count,
            &returned_algo_count,
            perf_results,
            workspace.get(),
            max_workspace_size));

    std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> algo_memory = {perf_results[0].algo, perf_results[0].memory};
    conv_bwd_data_algo_cache_map_[key] = algo_memory;
    return algo_memory;
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

    static const float kFloatZero = 0;
    static const float kFloatOne = 1;
    static const double kDoubleZero = 0;
    static const double kDoubleOne = 1;
    const void* zero{};
    const void* one{};
    if (x.dtype() == Dtype::kFloat64) {
        zero = &kDoubleZero;
        one = &kDoubleOne;
    } else {
        zero = &kFloatZero;
        one = &kFloatOne;
    }

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
            one,
            x_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(x_cont),
            filter_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(w_cont),
            conv_desc.get(),
            algo,
            workspace.get(),
            workspace_size,
            zero,
            y_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(y)));

    if (b) {
        assert(&b->device() == &x.device());
        assert(b->dtype() == x.dtype());

        int8_t ndim = x.ndim() - 2;
        assert(ndim > 0);

        Shape new_shape{};
        new_shape.emplace_back(1);
        new_shape.emplace_back(b->GetTotalSize());
        for (int8_t i = 0; i < ndim; ++i) {
            new_shape.emplace_back(1);
        }
        Array b_cont = AsContiguousArray(*b).Reshape(new_shape);
        std::shared_ptr<cudnnTensorStruct> b_desc = CreateTensorDescriptor(b_cont);
        CheckCudnnError(cudnnAddTensor(
                handle(),
                one,
                b_desc.get(),
                xchainer::internal::GetRawOffsetData<void>(b_cont),
                one,
                y_desc.get(),
                xchainer::internal::GetRawOffsetData<void>(y)));
    }
}

//    def convolution_backward_data(
//        core.ndarray W, core.ndarray x, core.ndarray b, core.ndarray y,
//        tuple pad, tuple stride, tuple dilation, int groups, *,
//        bint deterministic, bint auto_tune, str tensor_core):
//    cdef int dev_id = W.data.device.id
//    assert dev_id == x.data.device.id
//    assert dev_id == y.data.device.id
void Cudnn::ConvolutionBackwardData(
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

    CudaDevice& device = *static_cast<CudaDevice*>(&x.device());
    CudaBackend& backend = *static_cast<CudaBackend*>(&device.backend());

    // cdef float float_zero = 0, float_one = 1
    // cdef double double_zero = 0, double_one = 1
    // cdef size_t zero, one
    // if x.dtype == 'd':
    //     zero = <size_t>&double_zero
    //     one = <size_t>&double_one
    // else:
    //     zero = <size_t>&float_zero
    //     one = <size_t>&float_one

    static const float float_zero = 0;
    static const float float_one = 1;
    static const double double_zero = 0;
    static const double double_one = 1;
    const void* zero{};
    const void* one{};
    if (x.dtype() == Dtype::kFloat64) {
        zero = &double_zero;
        one = &double_one;
    } else {
        zero = &float_zero;
        one = &float_one;
    }

    // cdef bint use_tensor_core = _should_use_tensor_core(tensor_core, x.dtype)
    // cdef tuple conv_param = (pad, stride, x.dtype)

    // # cuDNN 7 supports dilation only in *_FWD_ALGO_IMPLICIT_GEMM, but
    // # it supports Tensor Cores only in *_FWD_ALGO_IMPLICIT_PRECOMP_GEMM.
    // if use_tensor_core:
    //     for i in dilation:
    //         if i > 1:
    //             use_tensor_core = False
    //             break

    // handle = get_handle()
    // x = core.ascontiguousarray(x)
    // W = core.ascontiguousarray(W)

    Array x_cont = AsContiguousArray(x);
    Array w_cont = AsContiguousArray(w);
    assert(y.IsContiguous());

    // # TODO(okuta) check performance
    // cdef size_t x_desc = cudnn.createTensorDescriptor()
    // cdef size_t y_desc = cudnn.createTensorDescriptor()
    // cdef size_t b_desc = cudnn.createTensorDescriptor()
    // cdef size_t filter_desc = cudnn.createFilterDescriptor()
    // cdef size_t conv_desc = cudnn.createConvolutionDescriptor()
    // cdef int algo
    // cdef size_t max_workspace_size = get_max_workspace_size()
    // cdef size_t workspace_size = 0
    // try:
    //     _create_tensor_nd_descriptor(x_desc, x, -1)
    //     _create_tensor_nd_descriptor(y_desc, y, -1)
    //     _create_filter_descriptor(filter_desc, W, cudnn.CUDNN_TENSOR_NCHW)
    //     _create_convolution_descriptor(
    //         conv_desc, pad, stride, dilation, groups, x.dtype,
    //         cudnn.CUDNN_CROSS_CORRELATION, use_tensor_core)
    std::shared_ptr<cudnnTensorStruct> x_desc = CreateTensorDescriptor(x_cont);
    std::shared_ptr<cudnnTensorStruct> y_desc = CreateTensorDescriptor(y);
    std::shared_ptr<cudnnFilterStruct> filter_desc = CreateFilterDescriptor(w_cont, CUDNN_TENSOR_NCHW);
    std::shared_ptr<cudnnConvolutionStruct> conv_desc =
            CreateConvolutionDescriptor(pad, stride, x.dtype(), CUDNN_CROSS_CORRELATION, dilation, groups);
    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();

    // if deterministic:
    //     algo = cudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
    //     workspace_size = cudnn.getConvolutionBackwardDataWorkspaceSize(
    //         handle, filter_desc, x_desc, conv_desc, y_desc, algo)
    //     # TODO(okuta): check workspace size
    // elif auto_tune and _cudnn_version >= 5000:
    //     algo, workspace_size = _find_algorithm_bwd_data(
    //         W, x, y, conv_param, handle, filter_desc, x_desc,
    //         conv_desc, y_desc, max_workspace_size)
    // else:
    //     algo, workspace_size = _get_algorithm_bwd_data(
    //         handle, filter_desc, x_desc, conv_desc, y_desc,
    //         max_workspace_size, use_tensor_core)
    // auto tune
    std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> algo_workspace_size =
            FindConvolutionBackwardDataAlgorithm(filter_desc, w_cont, x_desc, x_cont, conv_desc, y_desc, y, max_workspace_size, pad, stride);


    // max_workspace_size = max(max_workspace_size, workspace_size)
    // # TODO(okuta): allocate best size memory
    // workspace = memory.alloc(max_workspace_size)

    cudnnConvolutionBwdDataAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    // cudnn.convolutionBackwardData_v3(
    //     handle, one, filter_desc, W.data.ptr, x_desc, x.data.ptr,
    //     conv_desc, algo, workspace.ptr, max_workspace_size, zero, y_desc,
    //     y.data.ptr)
    CheckCudnnError(cudnnConvolutionBackwardData(
            handle(),
            one,
            filter_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(w_cont),
            x_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(x_cont),
            conv_desc.get(),
            algo,
            workspace.get(),
            workspace_size,
            zero,
            y_desc.get(),
            xchainer::internal::GetRawOffsetData<void>(y)));

    // if b is not None:
    //     assert dev_id == b.data.device.id
    //     ndim = x.ndim - 2
    //     b = core.ascontiguousarray(b).reshape((1, -1) + (1,) * ndim)
    //     _create_tensor_nd_descriptor(b_desc, b, -1)
    //     cudnn.addTensor_v3(handle, one, b_desc, b.data.ptr, one, y_desc,
    //                        y.data.ptr)
    if (b) {
        assert(&b->device() == &x.device());
        assert(b->dtype() == x.dtype());

        int8_t ndim = x.ndim() - 2;
        assert(ndim > 0);

        Shape new_shape{};
        new_shape.emplace_back(1);
        new_shape.emplace_back(b->GetTotalSize());
        for (int8_t i = 0; i < ndim; ++i) {
            new_shape.emplace_back(1);
        }
        Array b_cont = AsContiguousArray(*b).Reshape(new_shape);
        std::shared_ptr<cudnnTensorStruct> b_desc = CreateTensorDescriptor(b_cont);
        CheckCudnnError(cudnnAddTensor(
                handle(),
                one,
                b_desc.get(),
                xchainer::internal::GetRawOffsetData<void>(b_cont),
                one,
                y_desc.get(),
                xchainer::internal::GetRawOffsetData<void>(y)));
    }
}

}  // namespace internal
}  // namespace cuda
}  // namespace xchainer
