import unittest

import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing
from chainer import onnx_export


# @testing.parameterize([
#     {'layer': 'LinearFunction'},
#     {'layer': 'Reshape'},
#     {'layer': 'Convolution2DFunction'},
#     {'layer': 'AveragePooling2D'},
#     {'layer': 'MaxPooling2D'},
#     {'layer': 'BatchNormalization'},
#     {'layer': 'ReLU'},
#     {'layer': 'Softmax'},
#     {'layer': 'Add'},
# ])
class TestONNXExport(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.l1 = L.Convolution2D(None, 1, 1, 1, 0)
                    self.b2 = L.BatchNormalization(1)
                    self.l3 = L.Linear(None, 1)

            def __call__(self, x):
                h = F.relu(self.l1(x))
                h = self.b2(h)
                return self.l3(h)

        self.model = Model()

    def test_onnx_export_no_save(self):
        x = numpy.ones((1, 3, 7, 7)).astype(numpy.float32)
        model = onnx_export(self.model, x, None, True, 'test')
        assert model.graph.name == 'test'
        print(model)


testing.run_module(__name__, __file__)
