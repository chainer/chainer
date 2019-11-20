import unittest

import chainer
from chainer import testing
import numpy as np

from onnx_chainer_tests.helper import ONNXModelTest


@testing.parameterize(
    {'in_shape': (3, 5), 'name': 'softmax_cross_entropy'},
)
@unittest.skipUnless(
    int(chainer.__version__.split('.')[0]) >= 6,
    "SoftmaxCrossEntropy is supported from Chainer v6")
class TestSoftmaxCrossEntropy(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):
            def __init__(self):
                super(Model, self).__init__()

            def __call__(self, x, t):
                return chainer.functions.softmax_cross_entropy(x, t)

        self.model = Model()
        self.x = np.random.uniform(size=self.in_shape).astype('f')
        self.t = np.random.randint(size=self.in_shape[0], low=0,
                                   high=self.in_shape[1]).astype(np.int32)

    def test_output(self):
        self.expect(self.model, [self.x, self.t], name=self.name,
                    skip_opset_version=[7, 8])
