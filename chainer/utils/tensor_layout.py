from chainer.backends import cuda


def nchw_shape(my_shape, my_layout):
    if my_layout == 'NHWC':
        return (my_shape[0], my_shape[3], my_shape[1], my_shape[2])
    else:
        return my_shape


def my_shape(nchw_shape, my_layout):
    if my_layout == 'NHWC':
        return (nchw_shape[0], nchw_shape[2], nchw_shape[3], nchw_shape[1])
    else:
        return nchw_shape


def get_cudnn_tensor_layout(layout=None):
    if layout == 'NHWC':
        return cuda.cuda.cudnn.CUDNN_TENSOR_NHWC
    else:
        return cuda.cuda.cudnn.CUDNN_TENSOR_NCHW
