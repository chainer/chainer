import atexit

import functools
import numpy
import operator
import six

import cupy
from cupy import cuda
from cupy.cuda import cudnn


_cudnn_version = cudnn.getVersion()

_handles = {}


def get_handle():
    global _handles
    device = cuda.Device()
    handle = _handles.get(device.id, None)
    if handle is None:
        handle = cudnn.create()
        _handles[device.id] = handle
    return handle


@atexit.register
def reset_handles():
    global _handles
    handles = _handles
    _handles = {}

    for handle in six.itervalues(handles):
        cudnn.destroy(handle)


class Descriptor(object):

    def __init__(self, descriptor, destroyer):
        self.value = descriptor
        self.destroy = destroyer

    def __del__(self):
        if self.value:
            self.destroy(self.value)
            self.value = None


def get_data_type(dtype):
    if dtype.type == numpy.float32:
        return cudnn.CUDNN_DATA_FLOAT
    elif dtype.type == numpy.float64:
        return cudnn.CUDNN_DATA_DOUBLE
    elif dtype.type == numpy.float16:
        return cudnn.CUDNN_DATA_HALF
    else:
        raise TypeError('Dtype {} is not supported in cuDNN'.format(dtype))


def _to_ctypes_array(tup, dtype=numpy.intc):
    return numpy.array(tup, dtype=dtype).ctypes


def _succ_sublists(xs):
    # Returns successive sublists of xs.
    return [xs[i:] for i in six.moves.range(len(xs))]


def _compute_strides(shape):
    def aux(xs):
        return functools.reduce(operator.mul, xs[1:], 1)
    return tuple(map(aux, _succ_sublists(shape)))


def create_tensor_descriptor(arr, format=cudnn.CUDNN_TENSOR_NCHW):
    desc = Descriptor(cudnn.createTensorDescriptor(),
                      cudnn.destroyTensorDescriptor)
    if not arr.flags.c_contiguous:
        raise ValueError('cupy.cudnn supports c-contiguous arrays only')
    data_type = get_data_type(arr.dtype)
    if arr.ndim == 4:
        cudnn.setTensor4dDescriptor(desc.value, format, data_type, *arr.shape)
    else:
        c_shape = _to_ctypes_array(arr.shape)
        c_strides = _to_ctypes_array(_compute_strides(arr.shape))
        cudnn.setTensorNdDescriptor(desc.value, data_type, arr.ndim,
                                    c_shape.data, c_strides.data)
    return desc


def create_uninitialized_tensor_descriptor():
    """Create uninitialized tensor descriptor.

    Create a cudnnCreateTensorDescriptor_t that is not yet initialized.
    This is used by the batch normalization functions.
    """
    desc = Descriptor(cudnn.createTensorDescriptor(),
                      cudnn.destroyTensorDescriptor)
    return desc


def create_tensor_nd_descriptor(arr):
    desc = Descriptor(cudnn.createTensorDescriptor(),
                      cudnn.destroyTensorDescriptor)
    if not arr.flags.c_contiguous:
        raise ValueError('cupy.cudnn supports c-contiguous arrays only')
    data_type = get_data_type(arr.dtype)
    shape = arr.shape
    # numpy's stride is defined in bytes, but cudnn's stride is defined in
    # size of element
    strides = [s // arr.itemsize for s in arr.strides]

    c_shape = _to_ctypes_array(shape)
    c_strides = _to_ctypes_array(strides)
    cudnn.setTensorNdDescriptor(desc.value, data_type,
                                arr.ndim, c_shape.data, c_strides.data)

    return desc


def create_filter_descriptor(arr, mode=cudnn.CUDNN_CROSS_CORRELATION):
    desc = Descriptor(cudnn.createFilterDescriptor(),
                      cudnn.destroyFilterDescriptor)
    data_type = get_data_type(arr.dtype)
    if arr.ndim == 4:
        cudnn.setFilter4dDescriptor_v3(desc.value, data_type, *arr.shape)
    else:
        c_shape = _to_ctypes_array(arr.shape)
        cudnn.setFilterNdDescriptor_v3(desc.value, data_type, arr.ndim,
                                       c_shape.data)

    return desc


def create_convolution_descriptor(pad, stride, dtype,
                                  mode=cudnn.CUDNN_CROSS_CORRELATION):
    desc = Descriptor(cudnn.createConvolutionDescriptor(),
                      cudnn.destroyConvolutionDescriptor)
    ndim = len(pad)
    if ndim != len(stride):
        raise ValueError('pad and stride must be of same length')

    if ndim == 2:
        cudnn.setConvolution2dDescriptor(
            desc.value, pad[0], pad[1], stride[0], stride[1], 1, 1, mode)
    else:
        c_pad = _to_ctypes_array(pad)
        c_stride = _to_ctypes_array(stride)
        c_upscale = _to_ctypes_array((1,) * ndim)
        if _cudnn_version >= 3000:
            data_type = get_data_type(dtype)
            # TODO(takagi) Temporarily use computing precision of FP32 for
            #     storing precision of FP16.
            if dtype == numpy.float16:
                data_type = cudnn.CUDNN_DATA_FLOAT
            cudnn.setConvolutionNdDescriptor_v3(
                desc.value, ndim, c_pad.data, c_stride.data, c_upscale.data,
                mode, data_type)
        else:
            cudnn.setConvolutionNdDescriptor_v2(
                desc.value, ndim, c_pad.data, c_stride.data, c_upscale.data,
                mode)

    return desc


def create_pooling_descriptor(ksize, stride, pad, mode):
    desc = Descriptor(cudnn.createPoolingDescriptor(),
                      cudnn.destroyPoolingDescriptor)
    ndim = len(ksize)
    if ndim != len(stride) or ndim != len(pad):
        raise ValueError('ksize, stride, and pad must be of same length')

    if ndim == 2:
        cudnn.setPooling2dDescriptor_v3(
            desc.value, mode, ksize[0], ksize[1], pad[0], pad[1],
            stride[0], stride[1])
    else:
        c_ksize = _to_ctypes_array(ksize)
        c_pad = _to_ctypes_array(pad)
        c_stride = _to_ctypes_array(stride)
        cudnn.setPoolingNdDescriptor_v3(
            desc.value, mode, ndim, c_ksize.data, c_pad.data, c_stride.data)

    return desc


def _as4darray(arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    else:
        return arr.reshape(arr.shape[0], -1, 1, 1)


def activation_forward(x, mode):
    x = cupy.ascontiguousarray(x)
    y = cupy.empty_like(x)

    dtype = 'd' if x.dtype == 'd' else 'f'
    one = numpy.array(1, dtype=dtype).ctypes
    zero = numpy.array(0, dtype=dtype).ctypes
    handle = get_handle()
    x_mat = _as4darray(x)
    desc = create_tensor_descriptor(x_mat)
    cudnn.activationForward_v3(
        handle, mode, one.data, desc.value, x_mat.data.ptr,
        zero.data, desc.value, y.data.ptr)
    return y


def activation_backward(x, y, gy, mode):
    x = cupy.ascontiguousarray(x)
    gy = cupy.ascontiguousarray(gy)

    gx = cupy.empty_like(x)
    dtype = 'd' if x.dtype == 'd' else 'f'
    one = numpy.array(1, dtype=dtype).ctypes
    zero = numpy.array(0, dtype=dtype).ctypes
    handle = get_handle()
    y_mat = _as4darray(y)
    desc = create_tensor_descriptor(y_mat)
    cudnn.activationBackward_v3(
        handle, mode, one.data, desc.value, y.data.ptr,
        desc.value, gy.data.ptr, desc.value, x.data.ptr,
        zero.data, desc.value, gx.data.ptr)
    return gx


def create_dropout_descriptor(
        handle, dropout, states, state_size_in_bytes, seed):
    desc = Descriptor(cudnn.createDropoutDescriptor(),
                      cudnn.destroyDropoutDescriptor)
    cudnn.setDropoutDescriptor(desc.value, handle, dropout,
                               states, state_size_in_bytes, seed)
    return desc


def create_rnn_descriptor(hidden_size, num_layers, dropout_desc,
                          input_mode, direction, mode, data_type):
    desc = Descriptor(cudnn.createRNNDescriptor(),
                      cudnn.destroyRNNDescriptor)
    cudnn.setRNNDescriptor(
        desc.value, hidden_size, num_layers, dropout_desc.value,
        input_mode, direction, mode, data_type)
    return desc


def get_rnn_lin_layer_matrix_params(
        handle, rnn_desc, layer, x_desc, w_desc, w, lin_layer_id):
    mat_desc = Descriptor(cudnn.createFilterDescriptor(),
                          cudnn.destroyFilterDescriptor)
    ptr = numpy.array(0, dtype=numpy.intp)
    cudnn.getRNNLinLayerMatrixParams(
        handle, rnn_desc.value, layer, x_desc.value, w_desc.value, w.data.ptr,
        lin_layer_id, mat_desc.value, ptr.ctypes.data)
    offset = (ptr - w.data.ptr) // 4
    _, _, _, dim = cudnn.getFilterNdDescriptor(mat_desc.value, 3)
    size = numpy.prod(dim)
    mat = w[offset: offset + size]
    return mat


def get_rnn_lin_layer_bias_params(
        handle, rnn_desc, layer, x_desc, w_desc, w, lin_layer_id):
    bias_desc = Descriptor(cudnn.createFilterDescriptor(),
                           cudnn.destroyFilterDescriptor)
    ptr = numpy.array(0, dtype=numpy.intp)
    cudnn.getRNNLinLayerBiasParams(
        handle, rnn_desc.value, layer, x_desc.value, w_desc.value, w.data.ptr,
        lin_layer_id, bias_desc.value, ptr.ctypes.data)
    offset = (ptr - w.data.ptr) // 4
    _, _, _, dim = cudnn.getFilterNdDescriptor(bias_desc.value, 3)
    size = numpy.prod(dim)
    bias = w[offset: offset + size]
    return bias


def create_dropout_states(handle):
    state_size = cudnn.dropoutGetStatesSize(handle)
    return cupy.empty((state_size,), dtype='b')


if _cudnn_version >= 3000:
    def add_tensor(handle, alpha, biasDesc, biasData, beta, srcDestDesc,
                   srcDestData):
        cudnn.addTensor_v3(handle, alpha, biasDesc,
                           biasData, beta, srcDestDesc, srcDestData)
else:
    def add_tensor(handle, alpha, biasDesc, biasData, beta, srcDestDesc,
                   srcDestData):
        cudnn.addTensor_v2(handle, cudnn.CUDNN_ADD_SAME_C, alpha, biasDesc,
                           biasData, beta, srcDestDesc, srcDestData)
