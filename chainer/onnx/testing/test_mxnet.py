import collections
import os
import warnings

import numpy as np
import onnx

import chainer
import onnx_chainer

try:
    import mxnet
    MXNET_AVAILABLE = True
except ImportError:
    warnings.warn(
        'MXNet is not installed. Please install mxnet to use '
        'testing utility for compatibility checking.',
        ImportWarning)
    MXNET_AVAILABLE = False


def check_compatibility(model, x, fn, out_key='prob', opset_version=None):
    if opset_version is None:
        opset_version = onnx.defs.onnx_opset_version()
    if not MXNET_AVAILABLE:
        raise ImportError('check_compatibility requires MXNet.')

    chainer.config.train = False

    # Forward computation
    if isinstance(x, (list, tuple)):
        for i in x:
            assert isinstance(i, (np.ndarray, chainer.Variable))
        chainer_out = model(*x)
    elif isinstance(x, np.ndarray):
        chainer_out = model(chainer.Variable(x))
    elif isinstance(x, chainer.Variable):
        chainer_out = model(x)
    else:
        raise ValueError(
            'The \'x\' argument should be a list or tuple of numpy.ndarray or '
            'chainer.Variable, or simply numpy.ndarray or chainer.Variable '
            'itself. But a {} object was given.'.format(type(x)))

    if isinstance(chainer_out, (list, tuple)):
        chainer_out = [y.array for y in chainer_out]
    elif isinstance(chainer_out, dict):
        chainer_out = chainer_out[out_key]
        if isinstance(chainer_out, chainer.Variable):
            chainer_out = (chainer_out.array,)
    elif isinstance(chainer_out, chainer.Variable):
        chainer_out = (chainer_out.array,)
    else:
        raise ValueError('Unknown output type: {}'.format(type(chainer_out)))

    onnx_chainer.export(model, x, fn, opset_version=opset_version)

    sym, arg, aux = mxnet.contrib.onnx.import_model(fn)

    data_names = [graph_input for graph_input in sym.list_inputs()
                  if graph_input not in arg and graph_input not in aux]
    if len(data_names) > 1:
        data_shapes = [(n, x_.shape) for n, x_ in zip(data_names, x)]
    else:
        data_shapes = [(data_names[0], x.shape)]

    mod = mxnet.mod.Module(
        symbol=sym, data_names=data_names, context=mxnet.cpu(),
        label_names=None)
    mod.bind(
        for_training=False, data_shapes=data_shapes,
        label_shapes=None)
    mod.set_params(
        arg_params=arg, aux_params=aux, allow_missing=True,
        allow_extra=True)

    Batch = collections.namedtuple('Batch', ['data'])
    if isinstance(x, (list, tuple)):
        x = [mxnet.nd.array(x_.array) if isinstance(
            x_, chainer.Variable) else mxnet.nd.array(x_) for x_ in x]
    elif isinstance(x, chainer.Variable):
        x = [mxnet.nd.array(x.array)]
    elif isinstance(x, np.ndarray):
        x = [mxnet.nd.array(x)]

    mod.forward(Batch(x))
    mxnet_outs = mod.get_outputs()
    mxnet_out = [y.asnumpy() for y in mxnet_outs]

    for cy, my in zip(chainer_out, mxnet_out):
        np.testing.assert_almost_equal(cy, my, decimal=5)

    os.remove(fn)
