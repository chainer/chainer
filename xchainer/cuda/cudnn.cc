#include "xchainer/cuda/cudnn.h"

#include <memory>

#include <cudnn.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_device.h"
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
        CheckCudnnError(cudnnSetTensor4dDescriptor(desc, format, cudnn_dtype, n, c, h, w));
    } else {
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
        int64_t item_size = arr.item_size();
        StackVector<int, kMaxNdim> int_shape;
        StackVector<int, kMaxNdim> int_strides;
        for (int64_t v : arr.strides()) {
            int_strides.emplace_back(static_cast<int>(v / item_size));
        }
        for (int64_t v : arr.shape()) {
            int_shape.emplace_back(static_cast<int>(v));
        }
        CheckCudnnError(cudnnSetTensorNdDescriptor(desc, cudnn_dtype, arr.ndim(), &int_shape[0], &int_strides[0]));
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
        CheckCudnnError(cudnnSetFilter4dDescriptor(desc, cudnn_dtype, format, n, c, h, w));
    } else {
        StackVector<int, kMaxNdim> int_shape;
        for (int64_t v : arr.shape()) {
            int_shape.emplace_back(static_cast<int>(v));
        }
        CheckCudnnError(cudnnSetFilterNdDescriptor(desc, cudnn_dtype, format, arr.ndim(), &int_shape[0]));
    }
}

// cpdef _create_convolution_descriptor(
//         size_t desc, tuple pad, tuple stride, tuple dilation, int groups,
//         object dtype, int mode, bint use_tensor_core):
void SetConvolutionDescriptor(
        cudnnConvolutionDescriptor_t desc,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups,
        Dtype dtype,
        cudnnConvolutionMode_t mode) {
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

    StackVector<int, kMaxNdim> int_stride;
    StackVector<int, kMaxNdim> int_pad;
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
        StackVector<int, kMaxNdim> int_dilation;
        if (!dilation) {
            // TODO(sonots): Use assign(ndim, 1) if it becomes available
            for (decltype(ndim) i = 0; i < ndim; ++i) {
                int_dilation.emplace_back(1);
            }
        } else {
            for (int64_t v : *dilation) {
                int_dilation.emplace_back(static_cast<int>(v));
            }
        }
        CheckCudnnError(cudnnSetConvolutionNdDescriptor(desc, ndim, &int_pad[0], &int_stride[0], &int_dilation[0], mode, compute_type));
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

        CheckCudnnError(cudnnSetConvolution2dDescriptor(desc, p0, p1, s0, s1, d0, d1, mode, compute_type));
    }

    //     if _cudnn_version >= 7000:
    //         if use_tensor_core:
    //             math_type = cudnn.CUDNN_TENSOR_OP_MATH
    //             cudnn.setConvolutionMathType(desc, math_type)
    //         if groups > 1:
    //             cudnn.setConvolutionGroupCount(desc, groups)
    //     elif groups > 1:
    //         raise ValueError('groups must be one when cuDNN < 7.0')
    if (groups > 1) {
        CheckCudnnError(cudnnSetConvolutionGroupCount(desc, groups));
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
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        Dtype dtype,
        cudnnConvolutionMode_t mode,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups) {
    cudnnConvolutionDescriptor_t desc{};
    CheckCudnnError(cudnnCreateConvolutionDescriptor(&desc));
    auto shared_desc = std::shared_ptr<cudnnConvolutionStruct>{
            desc, [](cudnnConvolutionDescriptor_t desc) { CheckCudnnError(cudnnDestroyConvolutionDescriptor(desc)); }};
    SetConvolutionDescriptor(desc, pad, stride, dilation, groups, dtype, mode);
    return shared_desc;
}

// cpdef tuple _get_algorithm_fwd(
//         size_t handle, size_t x_desc, size_t filter_desc, size_t conv_desc,
//         size_t y_desc, size_t max_workspace_size, bint use_tensor_core):
//     algo = cudnn.getConvolutionForwardAlgorithm_v6(
//         handle, x_desc, filter_desc, conv_desc, y_desc,
//         cudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
//         max_workspace_size)
//     workspace_size = max_workspace_size
//     return algo, workspace_size
// std::pair<cudnnConvolutionFwdAlgo_t, size_t> GetConvolutionForwardAlgorithm(
//         cudnnHandle_t handle,
//         cudnnTensorDescriptor_t x_desc,
//         cudnnFilterDescriptor_t filter_desc,
//         cudnnConvolutionDescriptor_t conv_desc,
//         cudnnTensorDescriptor_t y_desc,
//         size_t max_workspace_size) {
//     cudnnConvolutionFwdAlgo_t algo{};
//     CheckCudnnError(cudnnGetConvolutionForwardAlgorithm(
//             handle, x_desc, filter_desc, conv_desc, y_desc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, max_workspace_size, &algo));
//     return {algo, max_workspace_size};
// }

// cpdef tuple _find_algorithm_fwd(
//         core.ndarray x, core.ndarray W, core.ndarray y, tuple conv_param,
//         size_t handle, size_t x_desc, size_t filter_desc, size_t conv_desc,
//         size_t y_desc, size_t max_workspace_size):
//     key = (x.data.device.id, x.shape, W.shape, y.shape, conv_param,
//            max_workspace_size)
//     if key in _algorithm_fwd:
//         return _algorithm_fwd[key]
//     workspace = memory.alloc(max_workspace_size)
//     ret = cudnn.findConvolutionForwardAlgorithmEx(
//         handle, x_desc, x.data.ptr, filter_desc, W.data.ptr, conv_desc, y_desc,
//         y.data.ptr, 1, workspace.ptr, max_workspace_size)
//     algo = (ret[0]['algo'], ret[0]['memory'])
//     _algorithm_fwd[key] = algo
//     return algo
// TODO(sonots): cache the result with key (device_id, x.shape, w,shape, y,shape, pad, stride, x.dtype, max_workspace_size)
std::pair<cudnnConvolutionFwdAlgo_t, size_t> FindConvolutionForwardAlgorithm(
        cudnnHandle_t handle,
        const std::shared_ptr<cudnnTensorStruct>& x_desc,
        const Array& x,
        const std::shared_ptr<cudnnFilterStruct>& filter_desc,
        const Array& w,
        const std::shared_ptr<cudnnConvolutionStruct>& conv_desc,
        const std::shared_ptr<cudnnTensorStruct>& y_desc,
        const Array& y,
        size_t max_workspace_size) {
    std::shared_ptr<void> workspace = y.device().Allocate(max_workspace_size);

    int requested_algo_count = 1;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(requested_algo_count);
    int returned_algo_count;

    CheckCudnnError(cudnnFindConvolutionForwardAlgorithmEx(
            handle,
            x_desc.get(),
            x.data().get(),
            filter_desc.get(),
            w.data().get(),
            conv_desc.get(),
            y_desc.get(),
            y.data().get(),
            requested_algo_count,
            &returned_algo_count,
            &perf_results[0],
            workspace.get(),
            max_workspace_size));

    return {perf_results[0].algo, perf_results[0].memory};
}

// def convolution_forward(
//         core.ndarray x, core.ndarray W, core.ndarray b, core.ndarray y,
//         tuple pad, tuple stride, tuple dilation, int groups, *,
//         bint auto_tune, str tensor_core):
// TODO(sonots): Support tensor core
void ConvolutionForward(
        CudaDevice& device,
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        Array& y,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups) {
    // cdef int dev_id = x.data.device.id
    // assert dev_id == W.data.device.id
    // assert dev_id == y.data.device.id
    assert(&device == &x.device());
    assert(&device == &y.device());
    assert(&device == &w.device());
    assert(x.dtype() == Dtype::kFloat32 || x.dtype() == Dtype::kFloat64);
    assert(y.dtype() == x.dtype());
    assert(w.dtype() == Dtype::kFloat32 || w.dtype() == Dtype::kFloat64);

    // cdef float float_zero = 0, float_one = 1
    // cdef double double_zero = 0, double_one = 1
    // cdef size_t zero, one
    // if x.dtype == 'd':
    //     zero = <size_t>&double_zero
    //     one = <size_t>&double_one
    // else:
    //     zero = <size_t>&float_zero
    //     one = <size_t>&float_one
    auto zero = x.dtype() == Dtype::kFloat64 ? double{0} : float{0};
    auto one = x.dtype() == Dtype::kFloat64 ? double{1} : float{1};

    // cdef bint use_tensor_core = _should_use_tensor_core(tensor_core, x.dtype)
    // cdef tuple conv_param = (pad, stride, x.dtype)
    //
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
    cudnnHandle_t handle = device.cudnn_handle();
    Array x_cont = AsContiguousArray(x);
    Array w_cont = AsContiguousArray(w);
    assert(y.IsContiguous());

    // # TODO(okuta) check performance
    // cdef size_t x_desc = cudnn.createTensorDescriptor()
    // cdef size_t y_desc = cudnn.createTensorDescriptor()
    // cdef size_t b_desc = cudnn.createTensorDescriptor()
    // cdef size_t filter_desc = cudnn.createFilterDescriptor()
    // cdef size_t conv_desc = cudnn.createConvolutionDescriptor()
    //
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
    size_t max_workspace_size = device.max_workspace_size();

    // if auto_tune and _cudnn_version >= 5000:
    //     algo, workspace_size = _find_algorithm_fwd(
    //         x, W, y, conv_param, handle, x_desc, filter_desc,
    //         conv_desc, y_desc, max_workspace_size)
    // else:
    //     algo, workspace_size = _get_algorithm_fwd(
    //         handle, x_desc, filter_desc, conv_desc, y_desc,
    //         max_workspace_size, use_tensor_core)
    // max_workspace_size = max(max_workspace_size, workspace_size)
    // # TODO(okuta): allocate best size memory
    // workspace = memory.alloc(max_workspace_size)

    // auto tune
    std::tuple<cudnnConvolutionFwdAlgo_t, size_t> algo_workspace_size =
            FindConvolutionForwardAlgorithm(handle, x_desc, x_cont, filter_desc, w_cont, conv_desc, y_desc, y, max_workspace_size);
    cudnnConvolutionFwdAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    // cudnn.convolutionForward(
    //     handle, one, x_desc, x.data.ptr, filter_desc, W.data.ptr,
    //     conv_desc, algo, workspace.ptr, max_workspace_size, zero, y_desc,
    //     y.data.ptr)
    CheckCudnnError(cudnnConvolutionForward(
            handle,
            &one,
            x_desc.get(),
            x_cont.data().get(),
            filter_desc.get(),
            w_cont.data().get(),
            conv_desc.get(),
            algo,
            workspace.get(),
            workspace_size,
            &zero,
            y_desc.get(),
            y.data().get()));

    // if b is not None:
    //     assert dev_id == b.data.device.id
    //     ndim = x.ndim - 2
    //     b = core.ascontiguousarray(b).reshape((1, -1) + (1,) * ndim)
    //     _create_tensor_nd_descriptor(b_desc, b, -1)
    //     cudnn.addTensor_v3(handle, one, b_desc,
    //                        b.data.ptr, one, y_desc, y.data.ptr)
    if (b) {
        assert(&device == &b->device());
        assert(b->dtype() == Dtype::kFloat32 || b->dtype() == Dtype::kFloat64);
        int8_t ndim = x.ndim() - 2;
        assert(ndim > 0);

        Shape new_shape;
        new_shape.emplace_back(1);
        new_shape.emplace_back(b->GetTotalSize());
        // TODO(sonots): Use assign(ndim, 1) if it becomes available
        for (int8_t i = 0; i < ndim; ++i) {
            new_shape.emplace_back(1);
        }
        Array b_cont = AsContiguousArray(*b).Reshape(new_shape);
        std::shared_ptr<cudnnTensorStruct> b_desc = CreateTensorDescriptor(b_cont);
        CheckCudnnError(cudnnAddTensor(handle, &one, b_desc.get(), b_cont.data().get(), &one, y_desc.get(), y.data().get()));
    }
}

}  // namespace cuda
}  // namespace xchainer
