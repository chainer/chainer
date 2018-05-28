import unittest

import numpy

import chainer
from chainer.exporters import caffe
import chainer.functions as F
import chainer.links as L
from chainer import testing


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
class TestCaffeExport(unittest.TestCase):

    def setUp(self):
        x = numpy.ones((1, 3, 7, 7)).astype(numpy.float32)
        self.x = chainer.Variable(x)

    def test_caffe_export_no_save(self):
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

        caffe.export(Model(), [self.x], None, True, 'test')

    def test_Reshape(self):
        class Link(chainer.Chain):

            def __call__(self, x):
                return F.reshape(x, (-1,))

        caffe.export(Link(), [self.x], None, True, 'test')

    def test_AveragePooling2D(self):
        class Link(chainer.Chain):
            def __call__(self, x):
                return F.average_pooling_2d(x, 1, 1, 0)

        caffe.export(Link(), [self.x], None, True, 'test')

    def test_MaxPooling2D(self):
        class Link(chainer.Chain):
            def __call__(self, x):
                return F.max_pooling_2d(x, 1, 1, 0)

        caffe.export(Link(), [self.x], None, True, 'test')

    def test_Softmax(self):
        class Link(chainer.Chain):
            def __call__(self, x):
                return F.softmax(x)

        caffe.export(Link(), [self.x], None, True, 'test')

    def test_Add(self):
        class Link(chainer.Chain):
            def __call__(self, x):
                return x + x

        caffe.export(Link(), [self.x], None, True, 'test')


testing.run_module(__name__, __file__)
