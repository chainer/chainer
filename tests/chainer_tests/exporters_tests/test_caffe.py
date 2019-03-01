import os
import unittest
import warnings

import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing


# The caffe submodule relies on protobuf which under protobuf==3.7.0 and
# Python 3.7 raises a DeprecationWarning from the collections module.
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    from chainer.exporters import caffe


# @testing.parameterize([
#     {'layer': 'LinearFunction'},
#     {'layer': 'Reshape'},
#     {'layer': 'Convolution2DFunction'},
#     {'layer': 'Deconvolution2DFunction'},
#     {'layer': 'AveragePooling2D'},
#     {'layer': 'MaxPooling2D'},
#     {'layer': 'BatchNormalization'},
#     {'layer': 'ReLU'},
#     {'layer': 'LeakyReLU'},
#     {'layer': 'Softmax'},
#     {'layer': 'Sigmoid'},
#     {'layer': 'Add'},
# ])
class TestCaffeExport(unittest.TestCase):

    def setUp(self):
        x = numpy.random.uniform(-1, 1, (1, 3, 7, 7)).astype(numpy.float32)
        self.x = chainer.Variable(x)

    def test_caffe_export_model(self):
        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.l1 = L.Convolution2D(None, 1, 1, 1, 0, groups=1)
                    self.b2 = L.BatchNormalization(1, eps=1e-2)
                    self.l3 = L.Deconvolution2D(None, 1, 1, 1, 0, groups=1)
                    self.l4 = L.Linear(None, 1)

            def forward(self, x):
                h = F.relu(self.l1(x))
                h = self.b2(h)
                h = self.l3(h)
                return self.l4(h)

        assert_export_import_match(Model(), self.x)

    def test_Reshape(self):
        class Link(chainer.Chain):

            def forward(self, x):
                return F.reshape(x, (-1,))

        assert_export_import_match(Link(), self.x)

    def test_LinearFunction(self):
        W = numpy.random.uniform(-1, 1, (1, numpy.prod(self.x.shape[1:])))

        class Link(chainer.Chain):
            def __call__(self, x):
                return F.linear(x, W)

        assert_export_import_match(Link(), self.x)

    def test_AveragePooling2D(self):
        class Link(chainer.Chain):
            def forward(self, x):
                return F.average_pooling_2d(x, 1, 1, 0)

        assert_export_import_match(Link(), self.x)

    def test_MaxPooling2D(self):
        class Link(chainer.Chain):
            def forward(self, x):
                return F.max_pooling_2d(x, 1, 1, 0)

        assert_export_import_match(Link(), self.x)

    def test_LeakyReLU(self):
        class Link(chainer.Chain):
            def __call__(self, x):
                return F.leaky_relu(x, slope=0.1)

        assert_export_import_match(Link(), self.x)

    def test_Softmax(self):
        class Link(chainer.Chain):
            def forward(self, x):
                return F.softmax(x)

        assert_export_import_match(Link(), self.x)

    def test_Sigmoid(self):
        class Link(chainer.Chain):
            def forward(self, x):
                return F.sigmoid(x)

        assert_export_import_match(Link(), self.x)

    def test_Add(self):
        class Link(chainer.Chain):
            def forward(self, x):
                return x + x

        assert_export_import_match(Link(), self.x)


def assert_export_import_match(l1, x):
    """Asserts that results from original Link and re-imported Link are close.

    """

    l2 = export_and_import(l1, (x,), True)
    inputs = {'data': x}
    outputs = [l2.layers[-1][0]]
    with chainer.using_config('train', False):
        for v1, v2 in zip(l1(x), l2(inputs=inputs, outputs=outputs)[0]):
            testing.assert_allclose(v1.data, v2.data)


def export_and_import(l, args, export_params=True, graph_name='test'):
    """Exports the given Link as Caffe model and returns the re-imported Link.

    """

    with chainer.utils.tempdir() as tempdir:
        caffe.export(l, args, tempdir, export_params, graph_name)

        prototxt = os.path.join(tempdir, 'chainer_model.prototxt')
        caffemodel = os.path.join(tempdir, 'chainer_model.caffemodel')

        assert os.path.exists(prototxt)
        assert os.path.exists(caffemodel)

        # with open(prototxt) as f: print(f.read())
        return L.caffe.CaffeFunction(caffemodel)


testing.run_module(__name__, __file__)
