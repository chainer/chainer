from collections import namedtuple
import tempfile
import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import mxnet as mx
import onnx_chainer
import onnx_mxnet

import pytest


class SmallCNN(chainer.Chain):

    def __init__(self):
        super(SmallCNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 10, 5)
            self.conv2 = L.Convolution2D(10, 20, 5)
            self.fc1 = L.Linear(None, 50)
            self.fc2 = L.Linear(50, 10)

    def __call__(self, x):
        x = F.relu(F.max_pooling_2d(self.conv1(x), 2))
        x = F.relu(F.max_pooling_2d(F.dropout(self.conv2(x)), 2))
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return x


def check_output(model, x):
    with tempfile.NamedTemporaryFile('wb') as fp:
        onnx_chainer.export(model, x, fp)

        sym, params = onnx_mxnet.import_model(fp.name)

        mod = mx.mod.Module(
            symbol=sym, data_names=['input_0'], context=mx.cpu(),
            label_names=None)
        mod.bind(
            for_training=False, data_shapes=[('input_0', x.shape)],
            label_shapes=None)
        mod.set_params(arg_params=params, aux_params=None, allow_missing=True)

        Batch = namedtuple('Batch', ['data'])
        mod.forward(Batch([mx.nd.array(x)]))

        mxnet_out = mod.get_outputs()[0].asnumpy()

        y = model(x)
        if isinstance(y, dict):
            y = y['prob']
        chainer_out = y.array

        np.testing.assert_almost_equal(
            chainer_out, mxnet_out, decimal=5)


class TestSmallCNN(unittest.TestCase):

    def setUp(self):

        self.model = SmallCNN()
        self.x = np.random.randn(3, 1, 28, 28).astype(np.float32)

    def test_export_test(self):
        chainer.config.train = False
        check_output(self.model, self.x)


class TestVGG16(unittest.TestCase):

    def setUp(self):

        self.model = L.VGG16Layers(None)
        self.x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    @pytest.mark.slow
    def test_export_test(self):
        chainer.config.train = False
        check_output(self.model, self.x)


class TestResNet50(unittest.TestCase):

    def setUp(self):

        self.model = L.ResNet50Layers(None)
        self.x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    @pytest.mark.slow
    def test_export_test(self):
        chainer.config.train = False
        check_output(self.model, self.x)
