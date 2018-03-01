import tempfile
import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from onnx_caffe2.backend import Caffe2Backend
from onnx_caffe2.backend import run_model
from onnx_caffe2.helper import benchmark_caffe2_model

import onnx
import onnx_chainer

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
        onnx_model = onnx.ModelProto.FromString(open(fp.name, 'rb').read())

        init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(
            onnx_model.graph, device='CPU')

        benchmark_caffe2_model(init_net, predict_net)

        y = model(x)
        if isinstance(y, dict):
            y = y['prob']
        chainer_out = y.array
        caffe2_out = run_model(onnx_model, [x])[0]

        np.testing.assert_almost_equal(
            chainer_out, caffe2_out, decimal=5)


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
