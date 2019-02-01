import unittest

import numpy as np
import onnx

import chainer
import chainer.links as L
from chainer import testing
from onnx_chainer.testing import test_onnxruntime


@testing.parameterize(
    # Convolution2D
    {'link': L.Convolution2D, 'in_shape': (1, 3, 5, 5), 'in_type': np.float32,
     'args': [None, 3, 3, 1, 1],
     'kwargs': {}},
    {'link': L.Convolution2D, 'in_shape': (1, 3, 5, 5), 'in_type': np.float32,
     'args': [None, 3, 3, 1, 1, True],
     'kwargs': {}},
    {'link': L.Convolution2D, 'in_shape': (1, 3, 5, 5), 'in_type': np.float32,
     'args': [None, 3, 3, 1, 1],
     'kwargs': {'groups': 3}},

    # ConvolutionND
    {'link': L.ConvolutionND, 'in_shape': (1, 2, 3, 5), 'in_type': np.float32,
     'args': [2, 2, 4, 3, 1, 0],
     'kwargs': {}},
    {'link': L.ConvolutionND, 'in_shape': (1, 2, 3, 5), 'in_type': np.float32,
     'args': [2, 2, 4, 3, 1, 0, True],
     'kwargs': {}},
    {'link': L.ConvolutionND, 'in_shape': (1, 3, 5, 5, 5),
     'in_type': np.float32,
     'args': [3, 3, 4, 3, 1, 0],
     'kwargs': {}},

    # DilatedConvolution2D
    {'link': L.DilatedConvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 3, 1, 1, 2],
     'kwargs': {}},
    {'link': L.DilatedConvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 3, 1, 1, 2, True],
     'kwargs': {}},

    # Deconvolution2D
    {'link': L.Deconvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 4, 2, 0],
     'kwargs': {}},
    {'link': L.Deconvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 4, 2, 0, True],
     'kwargs': {}},

    # EmbedID
    {'link': L.EmbedID, 'in_shape': (1, 10), 'in_type': np.int,
     'args': [5, 8],
     'kwargs': {}},

    # Linear
    {'link': L.Linear, 'in_shape': (1, 10), 'in_type': np.float32,
     'args': [None, 8],
     'kwargs': {}},
    {'link': L.Linear, 'in_shape': (1, 10), 'in_type': np.float32,
     'args': [None, 8, True],
     'kwargs': {}},
)
class TestConnections(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, link, args, kwargs):
                super(Model, self).__init__()
                with self.init_scope():
                    self.l1 = link(*args, **kwargs)

            def __call__(self, x):
                return self.l1(x)

        self.model = Model(self.link, self.args, self.kwargs)
        if self.link is L.EmbedID:
            self.x = np.random.randint(0, self.args[0], size=self.in_shape)
            self.x = self.x.astype(self.in_type)
        else:
            self.x = np.zeros(self.in_shape, dtype=self.in_type)
        self.fn = self.link.__name__ + '.onnx'

    def test_output(self):
        for opset_version in range(
                test_onnxruntime.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            test_onnxruntime.check_output(
                self.model, self.x, self.fn, opset_version=opset_version)
