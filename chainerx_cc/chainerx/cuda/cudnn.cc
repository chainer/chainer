#include "chainerx/cuda/cudnn.h"

#include <cudnn.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {

CudnnError::CudnnError(cudnnStatus_t status) : ChainerxError{cudnnGetErrorString(status)}, status_{status} {}

void CheckCudnnError(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw CudnnError{status};
    }
}

namespace {

cudnnDataType_t GetCudnnDataType(Dtype dtype) {
    switch (dtype) {
        case Dtype::kFloat16:
            return CUDNN_DATA_HALF;
        case Dtype::kFloat32:
            return CUDNN_DATA_FLOAT;
        case Dtype::kFloat64:
            return CUDNN_DATA_DOUBLE;
        default:
            throw DtypeError{"Dtype ", dtype, " is not supported in cuDNN"};
    }
}

template <typename T, typename U, typename... ErrorArgs>
T narrow(U u, const ErrorArgs&... error_args) {
    auto t = static_cast<T>(u);
    if (static_cast<U>(t) != u) {
        throw ChainerxError{error_args...};
    }
    return t;
}

template <typename T>
StackVector<int, kMaxNdim> GetIntStackVector(const T& container, const char* src) {
    StackVector<int, kMaxNdim> int_container;
    for (size_t i = 0; i < container.size(); ++i) {
        int_container.emplace_back(
                narrow<int>(container[i], "Casting the ", src, ": ", container[i], " at dimension: ", i, " to int failed."));
    }
    return int_container;
}

StackVector<int, kMaxNdim> GetIntShape(const Shape& shape) { return GetIntStackVector(shape, "shape size"); }

StackVector<int, kMaxNdim> GetIntKernelSize(const StackVector<int64_t, kMaxNdim>& kernel_size) {
    return GetIntStackVector(kernel_size, "kernel size");
}

StackVector<int, kMaxNdim> GetIntStride(const StackVector<int64_t, kMaxNdim>& stride) { return GetIntStackVector(stride, "stride"); }

StackVector<int, kMaxNdim> GetIntPad(const StackVector<int64_t, kMaxNdim>& pad) { return GetIntStackVector(pad, "pad"); }

StackVector<int, kMaxNdim> GetIntDilation(const StackVector<int64_t, kMaxNdim>& dilation) {
    return GetIntStackVector(dilation, "dilation");
}

// Returns strides divided by item size
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

namespace cuda_internal {

CudnnTensorDescriptor::CudnnTensorDescriptor() { CheckCudnnError(cudnnCreateTensorDescriptor(&desc_)); }

CudnnTensorDescriptor::~CudnnTensorDescriptor() {
    if (desc_ != nullptr) {
        CheckCudnnError(cudnnDestroyTensorDescriptor(desc_));
    }
}

CudnnTensorDescriptor::CudnnTensorDescriptor(const Array& arr) : CudnnTensorDescriptor{} {
    CHAINERX_ASSERT(arr.IsContiguous());

    cudnnDataType_t cudnn_dtype = GetCudnnDataType(arr.dtype());
    if (arr.shape().ndim() == 4) {
        StackVector<int, kMaxNdim> nchw = GetIntShape(arr.shape());
        CheckCudnnError(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, cudnn_dtype, nchw[0], nchw[1], nchw[2], nchw[3]));
    } else {
        StackVector<int, kMaxNdim> int_strides = GetIntArrayStrides(arr.strides(), arr.GetItemSize());  // strides divided by item size
        StackVector<int, kMaxNdim> int_shape = GetIntShape(arr.shape());
        CheckCudnnError(cudnnSetTensorNdDescriptor(desc_, cudnn_dtype, arr.ndim(), &int_shape[0], &int_strides[0]));
    }
}

CudnnFilterDescriptor::CudnnFilterDescriptor() { CheckCudnnError(cudnnCreateFilterDescriptor(&desc_)); }

CudnnFilterDescriptor::~CudnnFilterDescriptor() {
    if (desc_ != nullptr) {
        CheckCudnnError(cudnnDestroyFilterDescriptor(desc_));
    }
}

CudnnFilterDescriptor::CudnnFilterDescriptor(const Array& w) : CudnnFilterDescriptor{} {
    CHAINERX_ASSERT(w.IsContiguous());

    cudnnDataType_t cudnn_dtype = GetCudnnDataType(w.dtype());
    if (w.shape().ndim() == 4) {
        StackVector<int, kMaxNdim> nchw = GetIntShape(w.shape());
        CheckCudnnError(cudnnSetFilter4dDescriptor(desc_, cudnn_dtype, CUDNN_TENSOR_NCHW, nchw[0], nchw[1], nchw[2], nchw[3]));
    } else {
        StackVector<int, kMaxNdim> int_shape = GetIntShape(w.shape());
        CheckCudnnError(cudnnSetFilterNdDescriptor(desc_, cudnn_dtype, CUDNN_TENSOR_NCHW, w.ndim(), &int_shape[0]));
    }
}

CudnnConvolutionDescriptor::CudnnConvolutionDescriptor() { CheckCudnnError(cudnnCreateConvolutionDescriptor(&desc_)); }

CudnnConvolutionDescriptor::~CudnnConvolutionDescriptor() {
    if (desc_ != nullptr) {
        CheckCudnnError(cudnnDestroyConvolutionDescriptor(desc_));
    }
}

CudnnConvolutionDescriptor::CudnnConvolutionDescriptor(
        Dtype dtype,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups)
    : CudnnConvolutionDescriptor{} {
    size_t ndim = pad.size();
    CHAINERX_ASSERT(ndim == stride.size());
    CHAINERX_ASSERT(!dilation || ndim == dilation->size());

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

CudnnPoolingDescriptor::CudnnPoolingDescriptor() { CheckCudnnError(cudnnCreatePoolingDescriptor(&desc_)); }

CudnnPoolingDescriptor::~CudnnPoolingDescriptor() {
    if (desc_ != nullptr) {
        CheckCudnnError(cudnnDestroyPoolingDescriptor(desc_));
    }
}

CudnnPoolingDescriptor::CudnnPoolingDescriptor(
        cudnnPoolingMode_t mode,
        cudnnNanPropagation_t max_pooling_nan_opt,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride)
    : CudnnPoolingDescriptor{} {
    size_t ndim = kernel_size.size();
    CHAINERX_ASSERT(ndim == pad.size());
    CHAINERX_ASSERT(ndim == stride.size());

    StackVector<int, kMaxNdim> int_kernel_size = GetIntKernelSize(kernel_size);
    StackVector<int, kMaxNdim> int_pad = GetIntPad(pad);
    StackVector<int, kMaxNdim> int_stride = GetIntStride(stride);

    if (ndim == 2) {
        CheckCudnnError(cudnnSetPooling2dDescriptor(
                desc_,
                mode,
                max_pooling_nan_opt,
                int_kernel_size[0],
                int_kernel_size[1],
                int_pad[0],
                int_pad[1],
                int_stride[0],
                int_stride[1]));
    } else {
        CheckCudnnError(
                cudnnSetPoolingNdDescriptor(desc_, mode, max_pooling_nan_opt, ndim, &int_kernel_size[0], &int_pad[0], &int_stride[0]));
    }
}

CudnnHandle::~CudnnHandle() {
    if (handle_ != nullptr) {
        cudaSetDevice(device_index_);
        cudnnDestroy(handle_);
    }
}

cudnnHandle_t CudnnHandle::handle() {
    if (handle_ == nullptr) {
        CheckCudaError(cudaSetDevice(device_index_));
        CheckCudnnError(cudnnCreate(&handle_));
    }
    return handle_;
}

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
