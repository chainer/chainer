import chainer
import chainer.functions as F
from chainer import testing
from chainer.testing import attr
import numpy as np
import pytest

from onnx_chainer.testing import input_generator
from onnx_chainer_tests.helper import ONNXModelTest


@testing.parameterize(
    {'op_name': 'Neg', 'ops': '-a'},
    {'op_name': 'Absolute', 'ops': 'abs(a)'},
    {'op_name': 'Arccos', 'ops': 'chainer.functions.arccos(a)'},
    {'op_name': 'Arcsin', 'ops': 'chainer.functions.arcsin(a)'},
    {'op_name': 'Arctan', 'ops': 'chainer.functions.arctan(a)'},
    {'op_name': 'Cos', 'ops': 'chainer.functions.cos(a)'},
    {'op_name': 'Cosh', 'ops': 'chainer.functions.cosh(a)'},
    {'op_name': 'Clip', 'ops': 'chainer.functions.clip(a, 0.1, 0.2)'},
    {'op_name': 'Exp', 'ops': 'chainer.functions.exp(a)'},
    {'op_name': 'Sqrt', 'ops': 'chainer.functions.sqrt(a)'},
    {'op_name': 'RSqrt', 'ops': 'chainer.functions.rsqrt(a)'},
    {'op_name': 'PowVarConst',
     'ops': 'a ** 2.3'},
    {'op_name': 'PowConstVar',
     'ops': '2.3 ** a'},
    {'op_name': 'Sum', 'ops': 'chainer.functions.sum(a)'},
    {'op_name': 'Sum', 'ops': 'chainer.functions.sum(a, axis=1)',
     'condition': 'axis1'},
    {'op_name': 'Sum', 'ops': 'chainer.functions.sum(a, keepdims=True)',
     'condition': 'keepdims'},
    {'op_name': 'AddConstant', 'ops': 'a + 1'},
    {'op_name': 'MulConstant', 'ops': 'a * 2'},
    {'op_name': 'SubFromConstant', 'ops': '1 - a'},
    {'op_name': 'DivFromConstant', 'ops': '2 / a'},
    {'op_name': 'Max', 'ops': 'chainer.functions.max(a)'},
    {'op_name': 'Max', 'ops': 'chainer.functions.max(a, axis=0)',
     'condition': 'axis0'},
    {'op_name': 'Max', 'ops': 'chainer.functions.max(a, keepdims=True)',
     'condition': 'keepdims'},
    {'op_name': 'Mean', 'ops': 'chainer.functions.mean(a)'},
    {'op_name': 'Mean', 'ops': 'chainer.functions.mean(a, axis=0)',
     'condition': 'axis0'},
    {'op_name': 'Mean', 'ops': 'chainer.functions.mean(a, keepdims=True)',
     'condition': 'keepdims'},
    {'op_name': 'Min', 'ops': 'chainer.functions.min(a)'},
    {'op_name': 'Min', 'ops': 'chainer.functions.min(a, axis=0)',
     'condition': 'axis0'},
    {'op_name': 'Min', 'ops': 'chainer.functions.min(a, keepdims=True)',
     'condition': 'keepdims'},
    {'op_name': 'Prod', 'ops': 'chainer.functions.prod(a)'},
    {'op_name': 'Prod', 'ops': 'chainer.functions.prod(a, axis=0)',
     'condition': 'axis0'},
    {'op_name': 'Prod', 'ops': 'chainer.functions.prod(a, keepdims=True)',
     'condition': 'keepdims'},
    {'op_name': 'Log', 'ops': 'chainer.functions.log(a)'},
    {'op_name': 'LogSumExp', 'ops': 'chainer.functions.logsumexp(a)'},
    {'op_name': 'LogSumExp', 'ops': 'chainer.functions.logsumexp(a, axis=0)',
     'condition': 'axis0'},
    {'op_name': 'Sin', 'ops': 'chainer.functions.sin(a)'},
    {'op_name': 'Sinh', 'ops': 'chainer.functions.sinh(a)'},
    {'op_name': 'Square', 'ops': 'chainer.functions.square(a)'},
    {'op_name': 'Tan', 'ops': 'chainer.functions.tan(a)'},
    {'op_name': 'BroadcastTo',
     'ops': 'chainer.functions.broadcast_to(a, (2,2,3))'},
    {'op_name': 'Sign', 'ops': 'chainer.functions.sign(a)'},
)
class TestUnaryMathOperators(ONNXModelTest):

    @pytest.fixture(autouse=True)
    def setup_test_case(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a):
                if not isinstance(a, chainer.Variable):
                    a = chainer.Variable(a)
                return eval(self.ops)

        self.model = Model(self.ops)
        self.a = input_generator.positive_increasing(2, 3) / 10.0

        name = self.op_name.lower()
        if hasattr(self, 'condition'):
            name += '_' + self.condition
        self.name = name

        skip_opset_version = []
        if self.op_name == 'Cosh' or self.op_name == 'Sinh' or\
                self.op_name == 'Sign':
            skip_opset_version.append(7)
            skip_opset_version.append(8)

        if self.op_name == 'BroadcastTo':
            skip_opset_version.append(7)

        self.skip_opset_version = skip_opset_version

    def test_output(self):
        self.expect(self.model, self.a, name=self.name,
                    skip_opset_version=self.skip_opset_version,
                    expected_num_initializers=0)

    @attr.gpu
    def test_output_gpu(self):
        model, a = self.to_gpu(self.model, self.a)
        # test outputs are overwritten
        self.expect(model, a, name=self.name,
                    skip_opset_version=self.skip_opset_version,
                    expected_num_initializers=0)


@testing.parameterize(
    {'op_name': 'Add', 'ops': 'a + b'},
    {'op_name': 'Sub', 'ops': 'a - b'},
    {'op_name': 'Mul', 'ops': 'a * b'},
    {'op_name': 'Div', 'ops': 'a / b'},
    {'op_name': 'MatMul_transa',
     'ops': 'chainer.functions.matmul(a, b, transa=True)'},
    {'op_name': 'MatMul_transb',
     'ops': 'chainer.functions.matmul(a, b, transb=True)'},
    {'op_name': 'Maximum', 'ops': 'chainer.functions.maximum(a, b)'},
    {'op_name': 'Minimum', 'ops': 'chainer.functions.minimum(a, b)'},
    {'op_name': 'PowVarVar', 'ops': 'a ** b'},
)
class TestBinaryMathOperators(ONNXModelTest):

    def get_model(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a, b):
                return eval(self.ops)

        return Model(self.ops)

    def test_output(self):
        test_cases = ['matrix', 'vector']
        if not self.op_name.startswith('MatMul'):
            test_cases.append('scalar')

        for tc in test_cases:
            xs = getattr(self, tc)()
            name = '{}_{}'.format(self.op_name.lower(), tc)
            self.expect(
                self.get_model(), xs, name=name, expected_num_initializers=0)

    def matrix(self):
        a = chainer.Variable(input_generator.positive_increasing(5, 2, 3))
        b = chainer.Variable(input_generator.nonzero_increasing(5, 2, 3) * 0.3)
        return (a, b)

    def vector(self):
        a = chainer.Variable(input_generator.positive_increasing(2,))
        b = chainer.Variable(input_generator.nonzero_increasing(2,) * 0.3)
        return (a, b)

    def scalar(self):
        a = chainer.Variable(np.array(7, dtype=np.float32))
        b = chainer.Variable(np.array(2, dtype=np.float32))
        return (a, b)


@testing.parameterize(
    {'op_name': 'LinearInterpolate',
     'ops': 'chainer.functions.linear_interpolate(a, b, c)'},
)
class TestTernaryMathOperators(ONNXModelTest):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a, b, c):
                if not isinstance(a, chainer.Variable):
                    a = chainer.Variable(a)
                if not isinstance(b, chainer.Variable):
                    b = chainer.Variable(b)
                if not isinstance(c, chainer.Variable):
                    c = chainer.Variable(c)
                return eval(self.ops)

        self.model = Model(self.ops)
        a = chainer.Variable(input_generator.increasing(2, 3))
        b = chainer.Variable(input_generator.increasing(2, 3) * 0.3)
        c = chainer.Variable(input_generator.increasing(2, 3) * 0.7)
        self.x = (a, b, c)

    def test_output(self):
        name = self.op_name.lower()
        self.expect(self.model, self.x, name=name,
                    expected_num_initializers=0)


@testing.parameterize(
    {'op': 'argmax'},
    {'op': 'argmax', 'axis': 0},
    {'op': 'argmax', 'axis': 1},
    {'op': 'argmin'},
    {'op': 'argmin', 'axis': 0},
    {'op': 'argmin', 'axis': 1},
)
class TestArgMaxArgMin(ONNXModelTest):

    def test_output(self):
        class Model(chainer.Chain):

            def __init__(self, op, axis):
                self.axis = axis
                self.op = getattr(F, op)
                super(Model, self).__init__()

            def forward(self, x):
                return self.op(x, axis=self.axis)

        x = np.random.rand(2, 3).astype(np.float32)
        axis = getattr(self, 'axis', None)
        model = Model(self.op, axis)
        name = self.op
        if axis is not None:
            name += '_axis_{}'.format(self.axis)

        self.expect(model, x, name=name, expected_num_initializers=0)
