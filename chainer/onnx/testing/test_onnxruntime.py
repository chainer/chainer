import warnings

import numpy as np
import onnx

import chainer
import onnx_chainer

try:
    import onnxruntime as rt
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    warnings.warn(
        'ONNXRuntime is not installed. Please install it to use '
        ' the testing utility for ONNX-Chainer\'s converters.',
        ImportWarning)
    ONNXRUNTIME_AVAILABLE = False


MINIMUM_OPSET_VERSION = 7


def check_output(model, x, fn, out_key='prob', opset_version=None):
    if opset_version is None:
        opset_version = onnx.defs.onnx_opset_version()
    if not ONNXRUNTIME_AVAILABLE:
        raise ImportError('check_output requires onnxruntime.')

    chainer.config.train = False

    # Forward computation
    if isinstance(x, (list, tuple)):
        for i in x:
            assert isinstance(i, (np.ndarray, chainer.Variable))
        chainer_out = model(*x)
        x = tuple(
            _x.array if isinstance(_x, chainer.Variable) else _x for _x in x)
    elif isinstance(x, np.ndarray):
        chainer_out = model(chainer.Variable(x))
        x = (x,)
    elif isinstance(x, chainer.Variable):
        chainer_out = model(x)
        x = (x.array,)
    else:
        raise ValueError(
            'The \'x\' argument should be a list or tuple of numpy.ndarray or '
            'chainer.Variable, or simply a numpy.ndarray or a chainer.Variable'
            ' itself. But a {} object was given.'.format(type(x)))

    if isinstance(chainer_out, (list, tuple)):
        chainer_out = (y.array for y in chainer_out)
    elif isinstance(chainer_out, dict):
        chainer_out = chainer_out[out_key]
        if isinstance(chainer_out, chainer.Variable):
            chainer_out = (chainer_out.array,)
    elif isinstance(chainer_out, chainer.Variable):
        chainer_out = (chainer_out.array,)
    else:
        raise ValueError('Unknown output type: {}'.format(type(chainer_out)))

    onnx_model = onnx_chainer.export(model, x, fn, opset_version=opset_version)
    sess = rt.InferenceSession(onnx_model.SerializeToString())
    input_names = [i.name for i in sess.get_inputs()]
    rt_out = sess.run(
        None, {name: array for name, array in zip(input_names, x)})

    for cy, my in zip(chainer_out, rt_out):
        np.testing.assert_almost_equal(cy, my, decimal=5)
