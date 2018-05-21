#include "xchainer/cuda/cudnn.h"

#include <cudnn.h>

#include <memory>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"

namespace xchainer {
namespace cuda {

CudnnError::CudnnError(cudnnStatus_t status) : XchainerError{cudnnGetErrorString(status)}, status_{status} {}

void CheckCudnnError(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw CudnnError{status};
    }
}

namespace {

// cpdef int get_data_type(dtype) except? -1:
//     t = dtype.type
//     if t is numpy.float32:
//         return cudnn.CUDNN_DATA_FLOAT
//     elif t is numpy.float64:
//         return cudnn.CUDNN_DATA_DOUBLE
//     elif t is numpy.float16:
//         return cudnn.CUDNN_DATA_HALF
//     else:
//         raise TypeError('Dtype {} is not supported in cuDNN'.format(dtype)

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

// cpdef _create_tensor_nd_descriptor(
//        size_t desc, core.ndarray arr, int data_type=-1):
//    cdef vector.vector[int] c_shape, c_strides
//    cdef Py_ssize_t itemsize, s
//    if data_type == -1:  # `-1` is used instead of `None`
//        data_type = get_data_type(arr.dtype)
//    itemsize = arr.itemsize
//    for s in arr._strides:
//        c_strides.push_back(s // itemsize)
//    for s in arr._shape:
//        c_shape.push_back(s)
//    cudnn.setTensorNdDescriptor(
//        desc, data_type, arr.ndim, <size_t>&c_shape[0], <size_t>&c_strides[0])

void SetTensorNdDescriptor(cudnnTensorDescriptor_t desc, const Array& arr, cudnnDataType_t cudnn_dtype) {
    int64_t item_size = arr.item_size();
    std::vector<int> int_shape(arr.ndim());
    std::vector<int> int_strides(arr.ndim());
    for (int8_t i = 0; i < arr.ndim(); ++i) {
        int_strides.emplace_back(arr.strides()[i] / item_size);
    }
    for (int8_t i = 0; i < arr.ndim(); ++i) {
        int_shape.emplace_back(arr.shape()[i]);
    }
    cudnnSetTensorNdDescriptor(desc, cudnn_dtype, arr.ndim(), &int_shape[0], &int_strides[0]);
}

// cpdef _create_tensor_descriptor(size_t desc, core.ndarray arr, int format):
//    if not arr.flags.c_contiguous:
//        raise ValueError('cupy.cudnn supports c-contiguous arrays only')
//    data_type = get_data_type(arr.dtype)
//    if arr._shape.size() == 4:
//        n, c, h, w = arr.shape
//        data_type = get_data_type(arr.dtype)
//        cudnn.setTensor4dDescriptor(desc, format, data_type, n, c, h, w)
//    else:
//        _create_tensor_nd_descriptor(desc, arr)

void SetTensorDescriptor(cudnnTensorDescriptor_t desc, const Array& arr, cudnnTensorFormat_t format) {
    if (!arr.IsContiguous()) {
        throw XchainerError{"XChainer cuDNN supports only c-contiguous arrays"};
    }
    cudnnDataType_t cudnn_dtype = GetCudnnDataType(arr.dtype());
    if (arr.shape().ndim() == 4) {
        int n = static_cast<int>(arr.shape()[0]);
        int c = static_cast<int>(arr.shape()[1]);
        int h = static_cast<int>(arr.shape()[2]);
        int w = static_cast<int>(arr.shape()[3]);
        cudnnSetTensor4dDescriptor(desc, format, cudnn_dtype, n, c, h, w);
    } else {
        SetTensorNdDescriptor(desc, arr, cudnn_dtype);
    }
}

// cpdef _create_filter_descriptor(
//         size_t desc, core.ndarray arr, int format=cudnn.CUDNN_TENSOR_NCHW):
//     cdef vector.vector[int] c_shape
//     cdef Py_ssize_t s
//     data_type = get_data_type(arr.dtype)
//     if arr._shape.size() == 4:
//         n, c, h, w = arr.shape
//         cudnn.setFilter4dDescriptor_v4(
//             desc, data_type, format, n, c, h, w)
//     else:
//         for s in arr._shape:
//             c_shape.push_back(s)
//         cudnn.setFilterNdDescriptor_v4(
//             desc, data_type, format, arr.ndim, <size_t>&c_shape[0])

void SetFilterDescriptor(cudnnFilterDescriptor_t desc, const Array& arr, cudnnTensorFormat_t format) {
    cudnnDataType_t cudnn_dtype = GetCudnnDataType(arr.dtype());
    if (arr.shape().ndim() == 4) {
        int n = static_cast<int>(arr.shape()[0]);
        int c = static_cast<int>(arr.shape()[1]);
        int h = static_cast<int>(arr.shape()[2]);
        int w = static_cast<int>(arr.shape()[3]);
        cudnnSetFilter4dDescriptor(desc, cudnn_dtype, format, n, c, h, w);
    } else {
        std::vector<int> int_shape(arr.ndim());
        for (int8_t i = 0; i < arr.ndim(); ++i) {
            int_shape.emplace_back(arr.shape()[i]);
        }
        cudnnSetFilterNdDescriptor(desc, cudnn_dtype, format, arr.ndim(), &int_shape[0]);
    }
}

// cpdef _create_convolution_descriptor(
//         size_t desc, tuple pad, tuple stride, tuple dilation, int groups,
//         object dtype, int mode, bint use_tensor_core):
void SetConvolutionDescriptor(
        cudnnConvolutionDescriptor_t desc,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups,
        Dtype dtype,
        cudnnConvolutionMode_t mode,
        bool use_tensor_core) {
    //     cdef int d0, d1, p0, p1, s0, s1
    //     cdef vector.vector[int] c_pad, c_stride, c_dilation
    //     ndim = len(pad)
    //     if ndim != len(stride):
    //         raise ValueError('pad and stride must be of same length')
    //

    size_t ndim = pad.size();
    if (ndim != stride.size()) {
        throw DimensionError{"pad and stride must be of same length"};
    }
    if (dilation && ndim != dilation->size()) {
        throw DimensionError{"pad and dilation must be of same length"};
    }

    std::vector<int> int_stride(ndim);
    std::vector<int> int_pad(ndim);
    for (int64_t v : stride) {
        int_stride.emplace_back(static_cast<int>(v));
    }
    for (int64_t v : pad) {
        int_pad.emplace_back(static_cast<int>(v));
    }

    //     compute_type = get_data_type(dtype)
    //     # TODO(takagi) Temporarily use computing precision of FP32 for
    //     #     storing precision of FP16.
    //     if compute_type == cudnn.CUDNN_DATA_HALF:
    //         compute_type = cudnn.CUDNN_DATA_FLOAT
    //
    cudnnDataType_t compute_type = GetCudnnDataType(dtype);

    //     if ndim != 2:
    //         c_pad = pad
    //         c_stride = stride
    //         if dilation is None:
    //             c_dilation.assign(ndim, 1)
    //         else:
    //             c_dilation = dilation
    //             if _cudnn_version < 6000:
    //                 for i in c_dilation:
    //                     if i != 1:
    //                         raise ValueError(
    //                             'dilation must be one when cuDNN < 6.0')
    //         cudnn.setConvolutionNdDescriptor_v3(
    //             desc, ndim, <size_t>&c_pad[0], <size_t>&c_stride[0],
    //             <size_t>&c_dilation[0], mode, compute_type)

    if (ndim != 2) {
        std::vector<int> int_dilation(ndim);
        if (!dilation) {
            int_dilation.assign(ndim, 1);
        } else {
            for (int64_t v : *dilation) {
                int_dilation.emplace_back(static_cast<int>(v));
            }
        }
        cudnnSetConvolutionNdDescriptor(desc, ndim, &int_pad[0], &int_stride[0], &int_dilation[0], mode, compute_type);
    }

    //     else:
    //         if dilation is None:
    //             d0 = d1 = 1
    //         else:
    //             d0, d1 = dilation
    //         p0, p1 = pad
    //         s0, s1 = stride
    //         if _cudnn_version < 6000 and (d0 != 1 or d1 != 1):
    //             raise ValueError('dilation must be one when cuDNN < 6.0')
    //         if _cudnn_version >= 5000:
    //             cudnn.setConvolution2dDescriptor_v5(
    //                 desc, p0, p1, s0, s1, d0, d1, mode, compute_type)
    //         else:
    //             cudnn.setConvolution2dDescriptor_v4(
    //                 desc, p0, p1, s0, s1, 1, 1, mode)

    else {
        int d0 = 0, d1 = 0;
        if (!dilation) {
            d0 = d1 = 1;
        } else {
            d0 = static_cast<int>((*dilation)[0]);
            d1 = static_cast<int>((*dilation)[1]);
        }
        int p0 = static_cast<int>(pad[0]);
        int p1 = static_cast<int>(pad[1]);
        int s0 = static_cast<int>(stride[0]);
        int s1 = static_cast<int>(stride[1]);

        cudnnSetConvolution2dDescriptor(desc, p0, p1, s0, s1, d0, d1, mode, compute_type);
    }

    //     if _cudnn_version >= 7000:
    //         if use_tensor_core:
    //             math_type = cudnn.CUDNN_TENSOR_OP_MATH
    //             cudnn.setConvolutionMathType(desc, math_type)
    //         if groups > 1:
    //             cudnn.setConvolutionGroupCount(desc, groups)
    //     elif groups > 1:
    //         raise ValueError('groups must be one when cuDNN < 7.0')
    if (use_tensor_core) {
        cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
        cudnnSetConvolutionMathType(desc, math_type);
    }
    if (groups > 1) {
        cudnnSetConvolutionGroupCount(desc, groups);
    }
}

}  // namespace

// def create_tensor_descriptor(arr, format=cudnn.CUDNN_TENSOR_NCHW):
//    desc = Descriptor(cudnn.createTensorDescriptor(),
//                      py_cudnn.destroyTensorDescriptor)
//    _create_tensor_descriptor(desc.value, arr, format)
//    return desc

std::shared_ptr<cudnnTensorStruct> CreateTensorDescriptor(const Array& arr, cudnnTensorFormat_t format) {
    cudnnTensorDescriptor_t desc{};
    CheckCudnnError(cudnnCreateTensorDescriptor(&desc));
    auto shared_desc = std::shared_ptr<cudnnTensorStruct>{
            desc, [](cudnnTensorDescriptor_t desc) { CheckCudnnError(cudnnDestroyTensorDescriptor(desc)); }};
    SetTensorDescriptor(desc, arr, format);
    return shared_desc;
}

// def create_filter_descriptor(arr, format=cudnn.CUDNN_TENSOR_NCHW):
//     desc = Descriptor(cudnn.createFilterDescriptor(),
//                       py_cudnn.destroyFilterDescriptor)
//     _create_filter_descriptor(desc.value, arr, format)
//     return desc

std::shared_ptr<cudnnFilterStruct> CreateFilterDescriptor(const Array& arr, cudnnTensorFormat_t format) {
    cudnnFilterDescriptor_t desc{};
    CheckCudnnError(cudnnCreateFilterDescriptor(&desc));
    auto shared_desc = std::shared_ptr<cudnnFilterStruct>{
            desc, [](cudnnFilterDescriptor_t desc) { CheckCudnnError(cudnnDestroyFilterDescriptor(desc)); }};
    SetFilterDescriptor(desc, arr, format);
    return shared_desc;
}

// def create_convolution_descriptor(pad, stride, dtype,
//                                   mode=cudnn.CUDNN_CROSS_CORRELATION,
//                                   dilation=None,
//                                   use_tensor_core=False,
//                                   groups=1):
//     desc = Descriptor(cudnn.createConvolutionDescriptor(),
//                       py_cudnn.destroyConvolutionDescriptor)
//     _create_convolution_descriptor(
//         desc.value, pad, stride, dilation, groups,
//         dtype, mode, use_tensor_core)
//     return desc

std::shared_ptr<cudnnConvolutionStruct> CreateConvolutionDescriptor(
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        Dtype dtype,
        cudnnConvolutionMode_t mode,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        bool use_tensor_core,
        int groups) {
    cudnnConvolutionDescriptor_t desc{};
    CheckCudnnError(cudnnCreateConvolutionDescriptor(&desc));
    auto shared_desc = std::shared_ptr<cudnnConvolutionStruct>{
            desc, [](cudnnConvolutionDescriptor_t desc) { CheckCudnnError(cudnnDestroyConvolutionDescriptor(desc)); }};
    SetConvolutionDescriptor(desc, pad, stride, dilation, groups, dtype, mode, use_tensor_core);
    return shared_desc;
}

// cpdef tuple _get_algorithm_fwd(
//         size_t handle, size_t x_desc, size_t filter_desc, size_t conv_desc,
//         size_t y_desc, size_t max_workspace_size, bint use_tensor_core):
std::pair<cudnnConvolutionFwdAlgo_t, size_t> GetAlgorithmFwd(
        cudnnHandle_t handle,
        cudnnTensorDescriptor_t x_desc,
        cudnnFilterDescriptor_t filter_desc,
        cudnnConvolutionDescriptor_t conv_desc,
        cudnnTensorDescriptor_t y_desc,
        size_t max_workspace_size,
        bool use_tensor_core) {
    //     cdef int algo
    //     cdef workspace_size
    //     if use_tensor_core and _cudnn_version >= 7000:
    //         ret = cudnn.getConvolutionForwardAlgorithm_v7(
    //             handle, x_desc, filter_desc, conv_desc, y_desc, 10)
    //         for i in range(len(ret)):
    //             if ret[i]['memory'] <= max_workspace_size:
    //                 break
    //         else:
    //             raise RuntimeError('No conv fwd algo available with workspace size'
    //                                ' less equal {}'.format(max_workspace_size))
    //         if i != 0:
    //             msg = 'The best algo of conv fwd might not be selected due to '
    //                   'lack of workspace size ({})'.format(max_workspace_size)
    //             warnings.warn(msg)
    //         algo = ret[i]['algo']
    //         workspace_size = ret[i]['memory']
    if (use_tensor_core) {
        int requested_algo_count = 10;
        std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(requested_algo_count);
        int returned_algo_count;
        CheckCudnnError(cudnnGetConvolutionForwardAlgorithm_v7(
                handle, x_desc, filter_desc, conv_desc, y_desc, requested_algo_count, &returned_algo_count, &perf_results[0]));
        perf_results.resize(returned_algo_count);
        for (auto perf_result : perf_results) {
            if (perf_result.memory <= max_workspace_size) {
                return {perf_result.algo, perf_result.memory};
            }
        }
        throw XchainerError{"No conv fwd algo available with workspace size less equal ", max_workspace_size};
    }

    //     else:
    //         algo = cudnn.getConvolutionForwardAlgorithm_v6(
    //             handle, x_desc, filter_desc, conv_desc, y_desc,
    //             cudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
    //             max_workspace_size)
    //         workspace_size = max_workspace_size
    //     return algo, workspace_size

    cudnnConvolutionFwdAlgo_t algo{};
    CheckCudnnError(cudnnGetConvolutionForwardAlgorithm(
            handle, x_desc, filter_desc, conv_desc, y_desc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, max_workspace_size, &algo));
    return {algo, max_workspace_size};
}

}  // namespace cuda
}  // namespace xchainer
