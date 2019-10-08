import chainer
import chainer.functions as F
from chainer import testing
import numpy as np

from onnx_chainer_tests.helper import ONNXModelTest


@testing.parameterize(
    {'name': 'dropout', 'ops': lambda x: F.dropout(x, ratio=0.5)},
)
class TestNoises(ONNXModelTest):

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

    def test_output(self):
        self.expect(self.model, self.x, name=self.name)
