import warnings

import numpy
import six

from chainer import configuration
from chainer import functions
from chainer import initializer
from chainer import link
from chainer.links.caffe.protobuf3 import caffe_pb2 as caffe_pb
from chainer.links.connection import convolution_2d
from chainer.links.connection import deconvolution_2d
from chainer.links.connection import linear
from chainer.links.connection import scale
from chainer.links.normalization import batch_normalization
from chainer.utils import argument


try:
    # This method is undocumented, but is required to read large size of
    # model files when a user uses cpp-implementation.
    from google.protobuf.pyext import _message
    _message.SetAllowOversizeProtos(True)
except ImportError:
    pass

_type_to_method = {}
_oldname_to_method = {}


def _layer(typ, oldname):
    def decorator(meth):
        global _type_to_method
        _type_to_method[typ] = meth
        if oldname is not None:
            typevalue = getattr(caffe_pb.V1LayerParameter, oldname)
            _oldname_to_method[typevalue] = meth
        return meth
    return decorator


class _Blob(initializer.Initializer):

    chunk_size = 1024 * 1024

    def __init__(self, blob):
        super(_Blob, self).__init__()
        self.data = blob.data

    def __call__(self, array):
        array = array.ravel()
        size = len(array)
        indices = list(range(0, size, self.chunk_size))

        # Rather than accessing Protobuf's RepeatedScalar fields directly,
        # creating a intermediate list by indexing is more efficient due to
        # the implementation of the Python extension of Protobuf.
        # To avoid allocating excessively large lists, we limit the length
        # of lists by `chunk_size`.
        for start, end in zip(indices, indices[1:] + [size]):
            array[start:end] = self.data[start:end]


class _ConvolutionBlob(_Blob):

    def __init__(self, blob, group):
        super(_ConvolutionBlob, self).__init__(blob)
        self.group = group

    def __call__(self, array):
        n_out, n_in = array.shape[:2]

        part_out = n_out // self.group
        part_in = n_in // self.group

        array[...] = 0

        part_size = len(self.data) // self.group
        for i in six.moves.range(self.group):
            out_slice = slice(i * part_out, (i + 1) * part_out)
            in_slice = slice(i * part_in, (i + 1) * part_in)
            w = array[out_slice, in_slice]

            data = numpy.array(self.data[i * part_size:(i + 1) * part_size])
            w[:] = data.reshape(w.shape)


class CaffeFunction(link.Chain):

    """Caffe emulator based on the model file of Caffe.

    Given a protocol buffers file of a Caffe model, this class loads and
    emulates it on :class:`~chainer.Variable` objects. It supports the official
    reference models provided by BVLC.

    .. note::

       CaffeFunction ignores the following layers:

       - Layers that CaffeFunction does not support (including data layers)
       - Layers that have no top blobs
       - Layers whose bottom blobs are incomplete (i.e., some or all of them
         are not given nor computed)

    .. warning::

       It does not support full compatibility against Caffe. Some layers and
       configurations are not implemented in Chainer yet, though the reference
       models provided by the BVLC team are supported except data layers.

    .. admonition:: Example

       Consider we want to extract the (unnormalized) log class probability
       of given images using BVLC reference CaffeNet. The model can be
       downloaded from:

       http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

       We want to compute the ``fc8`` blob from the ``data`` blob. It is simply
       written as follows::

          # Load the model
          func = CaffeFunction('path/to/bvlc_reference_caffenet.caffemodel')

          # Minibatch of size 10
          x_data = numpy.ndarray((10, 3, 227, 227), dtype=numpy.float32)
          ...  # (Fill the minibatch here)

          # Forward the pre-trained net
          x = Variable(x_data)
          y, = func(inputs={'data': x}, outputs=['fc8'])

       The result ``y`` contains the Variable corresponding to the ``fc8``
       blob. The computational graph is memorized as a usual forward
       computation in Chainer, so we can run backprop through this pre-trained
       net.

    Args:
        model_path (str): Path to the binary-proto model file of Caffe.

    Attributes:
        forwards (dict): A mapping from layer names to corresponding functions.

    """

    def __init__(self, model_path):
        super(CaffeFunction, self).__init__()

        net = caffe_pb.NetParameter()
        with open(model_path, 'rb') as model_file:
            net.MergeFromString(model_file.read())

        self.forwards = {}
        self.split_map = {}
        self.layers = []

        if net.layer:
            for layer in net.layer:
                meth = _type_to_method.get(layer.type)
                if meth:
                    meth(self, layer)
                else:
                    warnings.warn(
                        'Skip the layer "%s", since CaffeFunction does not'
                        'support %s layer' % (layer.name, layer.type))
        else:  # v1 format
            for layer in net.layers:
                meth = _oldname_to_method.get(layer.type)
                if meth:
                    meth(self, layer)
                else:
                    warnings.warn(
                        'Skip the layer "%s", since CaffeFunction does not'
                        'support it' % layer.name)

    def forward(self, inputs, outputs, disable=(), **kwargs):
        """forward(self, inputs, outputs, disable=())

        Executes a sub-network of the network.

        This function acts as an interpreter of the network definition for
        Caffe. On execution, it interprets each layer one by one, and if the
        bottom blobs are already computed, then emulates the layer and stores
        output blobs as :class:`~chainer.Variable` objects.

        Args:
            inputs (dict): A dictionary whose key-value pairs indicate initial
                correspondences between blob names and
                :class:`~chainer.Variable` objects.
            outputs (Iterable): A list of blob names whose corresponding
                :class:`~chainer.Variable` objects are returned.
            disable (Iterable): A list of layer names that will be ignored
                during the forward computation.

        Returns:
            tuple: A tuple of output :class:`~chainer.Variable` objects
            corresponding to elements of the  `outputs` argument.

        """
        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs, train='train argument is not supported anymore. '
                'Use chainer.using_config')
            argument.assert_kwargs_empty(kwargs)

        variables = dict(inputs)
        disable = set(disable)
        for func_name, bottom, top in self.layers:
            if (func_name in disable or
                func_name not in self.forwards or
                    any(blob not in variables for blob in bottom)):
                continue

            func = self.forwards[func_name]
            input_vars = tuple(variables[blob] for blob in bottom)
            output_vars = func(*input_vars)
            if not isinstance(output_vars, (tuple, list)):
                output_vars = output_vars,
            for var, name in zip(output_vars, top):
                variables[name] = var

        self.variables = variables
        return tuple(variables[blob] for blob in outputs)

    def _add_layer(self, layer):
        bottom = []
        for blob_name in layer.bottom:
            bottom.append(self.split_map.get(blob_name, blob_name))
        self.layers.append((layer.name, bottom, list(layer.top)))

    @_layer('Concat', 'CONCAT')
    def _setup_concat(self, layer):
        param = layer.concat_param
        axis = param.axis
        if axis == 1 and param.concat_dim != 1:
            axis = param.concat_dim

        self.forwards[layer.name] = _ListArgumentFcuntion(
            functions.concat, axis=axis)
        self._add_layer(layer)

    @_layer('Convolution', 'CONVOLUTION')
    def _setup_convolution(self, layer):
        blobs = layer.blobs
        param = layer.convolution_param
        ksize = _get_ksize(param)
        stride = _get_stride(param)
        pad = _get_pad(param)
        num = _get_num(blobs[0])
        channels = _get_channels(blobs[0])
        bias_term = param.bias_term

        n_in = channels * param.group
        n_out = num

        func = convolution_2d.Convolution2D(
            n_in, n_out, ksize, stride, pad, nobias=not bias_term,
            initialW=_ConvolutionBlob(blobs[0], param.group),
            initial_bias=_Blob(blobs[1]) if bias_term else None)

        with self.init_scope():
            setattr(self, layer.name, func)
        self.forwards[layer.name] = _CallChildLink(self, layer.name)
        self._add_layer(layer)

    @_layer('Deconvolution', 'DECONVOLUTION')
    def _setup_deconvolution(self, layer):
        blobs = layer.blobs
        param = layer.convolution_param
        ksize = _get_ksize(param)
        stride = _get_stride(param)
        pad = _get_pad(param)
        num = _get_num(blobs[0])
        channels = _get_channels(blobs[0])
        bias_term = param.bias_term

        n_in = num
        n_out = channels * param.group

        func = deconvolution_2d.Deconvolution2D(
            n_in, n_out, ksize, stride, pad, nobias=not bias_term,
            initialW=_ConvolutionBlob(blobs[0], param.group),
            initial_bias=_Blob(blobs[1]) if bias_term else None)

        with self.init_scope():
            setattr(self, layer.name, func)
        self.forwards[layer.name] = _CallChildLink(self, layer.name)
        self._add_layer(layer)

    @_layer('Data', 'DATA')
    def _setup_data(self, layer):
        # We silently skip the data layer.
        pass

    @_layer('Dropout', 'DROPOUT')
    def _setup_dropout(self, layer):
        param = layer.dropout_param

        self.forwards[layer.name] = _SingleArgumentFunction(
            functions.dropout, ratio=param.dropout_ratio)
        self._add_layer(layer)

    @_layer('InnerProduct', 'INNER_PRODUCT')
    def _setup_inner_product(self, layer):
        param = layer.inner_product_param
        bias_term = param.bias_term
        if param.axis != 1:
            raise RuntimeError(
                'Non-default axis in InnerProduct is not supported')

        blobs = layer.blobs
        width, height = _get_width(blobs[0]), _get_height(blobs[0])

        func = linear.Linear(
            width, height, nobias=not bias_term,
            initialW=_Blob(blobs[0]),
            initial_bias=_Blob(blobs[1]) if bias_term else None)

        with self.init_scope():
            setattr(self, layer.name, func)
        self.forwards[layer.name] = _CallChildLink(self, layer.name)
        self._add_layer(layer)

    @_layer('LRN', 'LRN')
    def _setup_lrn(self, layer):
        param = layer.lrn_param
        if param.norm_region != param.ACROSS_CHANNELS:
            raise RuntimeError('Within-channel LRN is not supported')

        fwd = _SingleArgumentFunction(
            functions.local_response_normalization,
            n=param.local_size, k=param.k,
            alpha=param.alpha / param.local_size, beta=param.beta)
        self.forwards[layer.name] = fwd
        self._add_layer(layer)

    @_layer('Pooling', 'POOLING')
    def _setup_pooling(self, layer):
        param = layer.pooling_param
        ksize = _get_ksize(param)
        stride = _get_stride(param)
        pad = _get_pad(param)

        if param.pool == param.MAX:
            func = functions.max_pooling_2d
        elif param.pool == param.AVE:
            func = functions.average_pooling_2d
        else:
            raise RuntimeError('Stochastic pooling is not supported')

        if param.global_pooling and not ksize:
            # if global_pooling is set but no kernel size, the kernel size
            # is computed dynamically to cover the whole input feature map
            def _func(x, stride, pad):
                return func(x, x.shape[2:], stride=stride, pad=pad)
            fw = _SingleArgumentFunction(_func, stride=stride, pad=pad)
        else:
            fw = _SingleArgumentFunction(func, ksize, stride=stride, pad=pad)
        self.forwards[layer.name] = fw
        self._add_layer(layer)

    @_layer('ReLU', 'RELU')
    def _setup_relu(self, layer):
        slope = layer.relu_param.negative_slope

        if slope != 0:
            fw = _SingleArgumentFunction(functions.leaky_relu, slope=slope)
        else:
            fw = functions.relu

        self.forwards[layer.name] = fw
        self._add_layer(layer)

    @_layer('Reshape', None)
    def _setup_reshape(self, layer):
        shape = layer.reshape_param.shape.dim

        fw = _SingleArgumentFunction(functions.reshape, shape=shape)

        self.forwards[layer.name] = fw
        self._add_layer(layer)

    @_layer('BatchNorm', None)
    def _setup_batchnorm(self, layer):
        # Get layer parameters.
        blobs = layer.blobs
        param = layer.batch_norm_param
        use_global_stats = param.use_global_stats
        decay = param.moving_average_fraction
        eps = param.eps
        size = int(blobs[0].shape.dim[0])  # Get channel dim from mean blob.

        # Make BatchNormalization link.
        func = batch_normalization.BatchNormalization(
            size, decay=decay, eps=eps, use_gamma=False, use_beta=False)

        _Blob(blobs[0])(func.avg_mean)
        _Blob(blobs[1])(func.avg_var)

        # Scale the means and variances if a scaling factor is appended to the
        # blobs to correctly mimic to the behavior of Caffe. See
        # https://github.com/BVLC/caffe/issues/4885
        if len(blobs) >= 3:
            scaling_factor = blobs[2].data
            func.avg_mean /= scaling_factor[0]
            func.avg_var /= scaling_factor[0]

        with self.init_scope():
            setattr(self, layer.name, func)

        # Add layer.
        if use_global_stats:
            func_class = _SingleArgumentFunctionTestMode
        else:
            func_class = _SingleArgumentFunction
        fwd = func_class(_CallChildLink(self, layer.name), finetune=False)
        self.forwards[layer.name] = fwd
        self._add_layer(layer)

    @_layer('Eltwise', 'ELTWISE')
    def _setup_eltwise(self, layer):
        # stable_prod_grad parameter is not supported now.
        operation = layer.eltwise_param.operation
        coeffs = layer.eltwise_param.coeff or None
        self.forwards[layer.name] = _EltwiseFunction(operation, coeffs)
        self._add_layer(layer)

    @_layer('Scale', None)
    def _setup_scale(self, layer):
        # Following parameters are not supported now:
        # - negative axis
        # - num_axes
        # - filler
        # - bias_filler

        # Get layer parameters.
        bottom = layer.bottom
        blobs = layer.blobs
        axis = layer.scale_param.axis
        bias_term = layer.scale_param.bias_term

        # Case of only one bottom where W is learnt parameter.
        if len(bottom) == 1:
            W_shape = blobs[0].shape.dim
            func = scale.Scale(axis, W_shape, bias_term)
            _Blob(blobs[0])(func.W.data)
            if bias_term:
                _Blob(blobs[1])(func.bias.b.data)
        # Case of two bottoms where W is given as a bottom.
        else:
            shape = blobs[0].shape.dim if bias_term else None
            func = scale.Scale(
                axis, bias_term=bias_term, bias_shape=shape)
            if bias_term:
                _Blob(blobs[0])(func.bias.b.data)

        # Add layer.
        with self.init_scope():
            setattr(self, layer.name, func)
        self.forwards[layer.name] = _CallChildLink(self, layer.name)
        self._add_layer(layer)

    @_layer('Slice', 'SLICE')
    def _setup_slice(self, layer):
        if layer.slice_param.HasField('axis'):
            axis = layer.slice_param.axis
        elif layer.slice_param.HasField('slice_dim'):
            axis = layer.slice_param.slice_dim
        else:
            axis = 1

        if layer.slice_param.slice_point:
            indices_or_sections = list(layer.slice_param.slice_point)
        else:
            indices_or_sections = len(list(layer.top))

        self.forwards[layer.name] = _SingleArgumentFunction(
            functions.split_axis,
            indices_or_sections=indices_or_sections,
            axis=axis
        )

        self._add_layer(layer)

    @_layer('Softmax', 'SOFTMAX')
    def _setup_softmax(self, layer):
        if layer.softmax_param.axis != 1:
            raise RuntimeError(
                'Softmax along non-channel axis is not supported')

        if layer.softmax_param.engine == 0:  # DEFAULT
            fw = functions.softmax
        elif layer.softmax_param.engine == 1:  # CAFFE
            fw = _SingleArgumentFunctionWithCudnn(False, functions.softmax)
        elif layer.softmax_param.engine == 2:  # CUDNN
            fw = _SingleArgumentFunctionWithCudnn(True, functions.softmax)

        self.forwards[layer.name] = fw
        self._add_layer(layer)

    @_layer('Sigmoid', 'SIGMOID')
    def _setup_sigmoid(self, layer):
        if layer.sigmoid_param.engine == 0:  # DEFAULT
            fw = functions.sigmoid
        elif layer.sigmoid_param.engine == 1:  # CAFFE
            fw = _SingleArgumentFunctionWithCudnn(False, functions.sigmoid)
        elif layer.sigmoid_param.engine == 2:  # CUDNN
            fw = _SingleArgumentFunctionWithCudnn(True, functions.sigmoid)

        self.forwards[layer.name] = fw
        self._add_layer(layer)

    @_layer('SoftmaxWithLoss', 'SOFTMAX_LOSS')
    def _setup_softmax_with_loss(self, layer):
        if layer.softmax_param.axis != 1:
            raise RuntimeError(
                'Softmax along non-channel axis is not supported')

        self.forwards[layer.name] = functions.softmax_cross_entropy
        self._add_layer(layer)

    @_layer('Split', 'SPLIT')
    def _setup_split(self, layer):
        for top in layer.top:
            self.split_map[top] = layer.bottom[0]


# Internal functions

def _get_ksize(param):
    if param.kernel_h > 0:
        return param.kernel_h, param.kernel_w
    elif type(param.kernel_size) == int:
        return param.kernel_size
    elif len(param.kernel_size) == 1:
        return param.kernel_size[0]
    else:
        return param.kernel_size


def _get_stride(param):
    if param.stride_h > 0:
        return param.stride_h, param.stride_w
    elif type(param.stride) == int:
        return param.stride
    elif len(param.stride) == 0:
        return 1
    elif len(param.stride) == 1:
        return param.stride[0]
    else:
        return param.stride


def _get_pad(param):
    if param.pad_h > 0 or param.pad_w > 0:
        return param.pad_h, param.pad_w
    elif type(param.pad) == int:
        return param.pad
    elif len(param.pad) == 0:
        return 0
    elif len(param.pad) == 1:
        return param.pad[0]
    else:
        return param.pad


def _get_num(blob):
    if blob.num > 0:
        return blob.num
    else:
        return blob.shape.dim[0]


def _get_channels(blob):
    if blob.channels > 0:
        return blob.channels
    else:
        return blob.shape.dim[1]


def _get_height(blob):
    if blob.height > 0:
        return blob.height
    elif len(blob.shape.dim) == 2:
        return blob.shape.dim[0]
    elif len(blob.shape.dim) == 4:
        return blob.shape.dim[2]
    else:
        raise RuntimeError(
            '{}-dimentional array is not supported'.format(
                len(blob.shape.dim)))


def _get_width(blob):
    if blob.width > 0:
        return blob.width
    elif len(blob.shape.dim) == 2:
        return blob.shape.dim[1]
    elif len(blob.shape.dim) == 4:
        return blob.shape.dim[3]
    else:
        raise RuntimeError(
            '{}-dimentional array is not supported'.format(
                len(blob.shape.dim)))


# Internal class
# __call__ must return Variable or tuple

class _SingleArgumentFunction(object):

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.func(x, *self.args, **self.kwargs)


class _SingleArgumentFunctionTestMode(_SingleArgumentFunction):

    def __call__(self, x):
        with configuration.using_config('train', False):
            return super(_SingleArgumentFunctionTestMode, self).__call__(x)


class _ListArgumentFcuntion(object):

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def __call__(self, *xs):
        return self.func(xs, **self.kwargs)


class _SingleArgumentFunctionWithCudnn(_SingleArgumentFunction):

    def __init__(self, use_cudnn, func, *args, **kwargs):
        super(_SingleArgumentFunctionWithCudnn, self).__init__(
            func, *args, **kwargs)
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        with configuration.using_config('use_cudnn', self.use_cudnn):
            return super(_SingleArgumentFunctionWithCudnn, self).__call__(x)


class _CallChildLink(object):

    def __init__(self, caffe_func, name):
        self.name = name
        self.caffe_func = caffe_func

    def __call__(self, *xs, **kwargs):
        return self.caffe_func[self.name](*xs, **kwargs)


class _EltwiseFunction(object):

    def __init__(self, operation, coeffs=None):
        if coeffs is not None:
            assert len(coeffs) > 0
        self.operation = operation
        self.coeffs = coeffs

    def __call__(self, *xs):
        operation = self.operation

        if operation == 0:      # PROD
            return six.moves.reduce(lambda x, y: x * y, xs),

        elif operation == 1:    # SUM
            coeffs = self.coeffs
            if coeffs is not None:
                assert len(xs) == len(coeffs)
                xs = [x * coeff for x, coeff in zip(xs, coeffs)]
            return six.moves.reduce(lambda x, y: x + y, xs),

        elif operation == 2:    # MAX
            return six.moves.reduce(lambda x, y: functions.maximum(x, y), xs),

        else:
            raise ValueError('Invalid EltwiseParameter.EltwiseOp value.')
