import unittest

import numpy as np
import onnx

import chainer
import chainer.functions as F
from chainer import testing
from onnx_chainer.testing import test_onnxruntime


@testing.parameterize(
    {'name': 'dropout', 'ops': lambda x: F.dropout(x, ratio=0.5)},
)
class TestNoises(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, x):
                with chainer.using_config('train', True):
                    y = self.ops(x)
                return y

        self.model = Model(self.ops)
        self.x = np.zeros((1, 5), dtype=np.float32)
        self.fn = self.name + '.onnx'

    def test_output(self):
        for opset_version in range(
                test_onnxruntime.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            test_onnxruntime.check_output(
                self.model, self.x, self.fn, opset_version=opset_version)
