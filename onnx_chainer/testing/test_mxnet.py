import collections
import os
import warnings

import chainer
import numpy as np

from onnx_chainer.testing.test_onnxruntime import load_test_data

try:
    import mxnet
    MXNET_AVAILABLE = True
except ImportError:
    warnings.warn(
        'MXNet is not installed. Please install mxnet to use '
        'testing utility for compatibility checking.',
        ImportWarning)
    MXNET_AVAILABLE = False


def check_model_expect(test_path, input_names=None, rtol=1e-5, atol=1e-5):
    if not MXNET_AVAILABLE:
        raise ImportError('MXNet is not found on checking module.')

    model_path = os.path.join(test_path, 'model.onnx')
    sym, arg, aux = mxnet.contrib.onnx.import_model(model_path)

    mx_input_names = [graph_input for graph_input in sym.list_inputs()
                      if graph_input not in arg and graph_input not in aux]
    if input_names is not None:
        assert list(sorted(input_names)) == list(sorted(mx_input_names))

    test_data_sets = sorted([
        p for p in os.listdir(test_path) if p.startswith('test_data_set_')])
    for test_data in test_data_sets:
        test_data_path = os.path.join(test_path, test_data)
        assert os.path.isdir(test_data_path)
        inputs, outputs = load_test_data(
            test_data_path, mx_input_names, sym.list_outputs())

        data_shapes = [(name, array.shape) for name, array in inputs.items()]
        mod = mxnet.mod.Module(
            symbol=sym, data_names=mx_input_names, context=mxnet.cpu(),
            label_names=None)
        mod.bind(
            for_training=chainer.config.train,
            data_shapes=data_shapes, label_shapes=None)
        mod.set_params(
            arg_params=arg, aux_params=aux, allow_missing=True,
            allow_extra=True)
        Batch = collections.namedtuple('Batch', ['data'])
        mx_input = [mxnet.nd.array(array) for array in inputs.values()]
        mod.forward(Batch(mx_input))
        mx_outputs = [y.asnumpy() for y in mod.get_outputs()]
        for cy, my in zip(outputs.values(), mx_outputs):
            np.testing.assert_allclose(cy, my, rtol=rtol, atol=atol)
