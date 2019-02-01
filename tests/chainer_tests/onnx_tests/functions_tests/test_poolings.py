import unittest

import numpy as np
import onnx

import chainer
import chainer.functions as F
from chainer import testing
from onnx_chainer.testing import test_onnxruntime


@testing.parameterize(
    {'name': 'average_pooling_2d', 'ops': F.average_pooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [2, 1, 0], 'cover_all': None},
    {'name': 'average_pooling_2d', 'ops': F.average_pooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [3, 2, 1], 'cover_all': None},
    {'name': 'average_pooling_nd', 'ops': F.average_pooling_nd,
     'in_shape': (1, 3, 6, 6, 6), 'args': [2, 1, 1], 'cover_all': None},
    {'name': 'max_pooling_2d', 'ops': F.max_pooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [2, 1, 1], 'cover_all': False},
    {'name': 'max_pooling_2d', 'ops': F.max_pooling_2d,
     'in_shape': (1, 3, 6, 5), 'args': [3, 1, 1], 'cover_all': True},
    {'name': 'max_pooling_nd', 'ops': F.max_pooling_nd,
     'in_shape': (1, 3, 6, 6, 6), 'args': [2, 1, 1], 'cover_all': False},
    {'name': 'max_pooling_nd', 'ops': F.max_pooling_nd,
     'in_shape': (1, 3, 6, 5, 4), 'args': [3, 1, 1], 'cover_all': True},
)
class TestPoolings(unittest.TestCase):

    def setUp(self):
        ops = getattr(F, self.name)
        self.model = Model(ops, self.args, self.cover_all)
        self.x = np.ones(self.in_shape, dtype=np.float32)
        self.fn = self.name + '.onnx'

    def test_output(self):
        for opset_version in range(
                test_onnxruntime.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            test_onnxruntime.check_output(
                self.model, self.x, self.fn, opset_version=opset_version)


@testing.parameterize(
    {'name': 'max_pooling_2d', 'ops': F.max_pooling_2d,
     'in_shape': (1, 3, 6, 5), 'args': [2, 2, 1], 'cover_all': True},
    {'name': 'max_pooling_nd', 'ops': F.max_pooling_nd,
     'in_shape': (1, 3, 6, 5, 4), 'args': [2, 2, 1], 'cover_all': True},
)
class TestPoolingsWithUnsupportedSettings(unittest.TestCase):

    def setUp(self):
        ops = getattr(F, self.name)
        self.model = Model(ops, self.args, self.cover_all)
        self.x = np.ones(self.in_shape, dtype=np.float32)
        self.fn = self.name + '.onnx'

    def test_output(self):
        for opset_version in range(
                test_onnxruntime.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            with self.assertRaises(RuntimeError):
                test_onnxruntime.check_output(
                    self.model, self.x, self.fn, opset_version=opset_version)


class Model(chainer.Chain):

    def __init__(self, ops, args, cover_all):
        super(Model, self).__init__()
        self.ops = ops
        self.args = args
        self.cover_all = cover_all

    def __call__(self, x):
        if self.cover_all is not None:
            return self.ops(*([x] + self.args), cover_all=self.cover_all)
        else:
            return self.ops(*([x] + self.args))
