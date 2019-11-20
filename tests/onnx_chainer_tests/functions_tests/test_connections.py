import chainer
import chainer.links as L
from chainer import testing
import numpy as np

from onnx_chainer.testing import input_generator
from onnx_chainer_tests.helper import ONNXModelTest


@testing.parameterize(
    # Convolution2D
    {'link': L.Convolution2D, 'in_shape': (1, 3, 5, 5), 'in_type': np.float32,
     'args': [None, 3, 3, 1, 1],
     'kwargs': {}},
    {'link': L.Convolution2D, 'in_shape': (1, 3, 5, 5), 'in_type': np.float32,
     'args': [None, 3, 3, 1, 2, True],
     'kwargs': {}, 'name': 'Convolution2D_pad2_bias'},
    {'link': L.Convolution2D, 'in_shape': (1, 3, 5, 5), 'in_type': np.float32,
     'args': [None, 3, 3, 1, 1],
     'kwargs': {'groups': 3}, 'name': 'Convolution2D_groups3'},

    # ConvolutionND
    {'link': L.ConvolutionND, 'in_shape': (1, 2, 3, 5), 'in_type': np.float32,
     'args': [2, 2, 4, 3, 1, 0],
     'kwargs': {}},
    {'link': L.ConvolutionND, 'in_shape': (1, 2, 3, 5), 'in_type': np.float32,
     'args': [2, 2, 4, 3, 1, 0, True],
     'kwargs': {}, 'name': 'ConvolutionND_bias'},
    {'link': L.ConvolutionND, 'in_shape': (1, 3, 5, 5, 5),
     'in_type': np.float32,
     'args': [3, 3, 4, 3, 1, 0],
     'kwargs': {}, 'name': 'ConvolutionND_ndim3'},
    {'link': L.ConvolutionND, 'in_shape': (1, 6, 5, 5, 5),
     'in_type': np.float32, 'args': [3, 6, 4, 3, 1, 0],
     'kwargs': {'groups': 2}, 'name': 'ConvolutionND_group2'},

    # DilatedConvolution2D
    {'link': L.DilatedConvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 3, 1, 1, 2],
     'kwargs': {}},
    {'link': L.DilatedConvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 3, 1, 1, 2, True],
     'kwargs': {}, 'name': 'DilatedConvolution2D_bias'},

    # Deconvolution2D
    {'link': L.Deconvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 4, 2, 0],
     'kwargs': {}},
    {'link': L.Deconvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 4, 2, 0, True],
     'kwargs': {}, 'name': 'Deconvolution2D_bias'},
    {'link': L.Deconvolution2D, 'in_shape': (1, 4, 5, 5),
     'in_type': np.float32, 'args': [None, 6, 2, 2, 0, True],
     'kwargs': {'groups': 2}, 'name': 'Deconvolution2D_group3'},

    # DeconvolutionND
    # NOTE(disktnk): ONNX runtime accepts only 4-dimensional input X
    {'link': L.DeconvolutionND, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [2, 3, 3, 2, 2, 0, True],
     'kwargs': {}, 'name': 'DeconvolutionND'},
    {'link': L.DeconvolutionND, 'in_shape': (1, 6, 5, 5),
     'in_type': np.float32, 'args': [2, 6, 4, 2, 2, 0, True],
     'kwargs': {'groups': 2}, 'name': 'DeconvolutionND_group3'},

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
     'kwargs': {}, 'name': 'Linear_bias'},
)
class TestConnections(ONNXModelTest):

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
            self.x = input_generator.increasing(
                *self.in_shape, dtype=self.in_type)

    def test_output(self):
        name = self.link.__name__.lower()
        if hasattr(self, 'name'):
            name = self.name.lower()
        self.expect(self.model, self.x, name=name)
