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
    std::vector<int> int_shape{arr.ndim()};
    std::vector<int> int_strides{arr.ndim()};
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

}  // namespace cuda
}  // namespace xchainer
