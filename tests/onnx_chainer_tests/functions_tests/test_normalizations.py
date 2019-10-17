import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing
import numpy as np

from onnx_chainer.testing import input_generator
from onnx_chainer_tests.helper import get_initializer_names
from onnx_chainer_tests.helper import ONNXModelTest


@testing.parameterize(
    {
        'name': 'local_response_normalization',
        'input_argname': 'x',
        'args': {'k': 1, 'n': 3, 'alpha': 1e-4, 'beta': 0.75},
        'opset_version': 1
    },
    {
        'name': 'normalize',
        'input_argname': 'x',
        'args': {'axis': 1},
        'opset_version': 1
    }
)
class TestNormalizations(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args, input_argname):
                super(Model, self).__init__()
                self.ops = ops
                self.args = args
                self.input_argname = input_argname

            def __call__(self, x):
                self.args[self.input_argname] = x
                return self.ops(**self.args)

        ops = getattr(F, self.name)
        self.model = Model(ops, self.args, self.input_argname)
        self.x = input_generator.increasing(2, 5, 3, 3)

    def test_output(self):
        self.expect(self.model, self.x, name=self.name)


@testing.parameterize(
    {'kwargs': {}},
    {'kwargs': {'use_beta': False}, 'condition': 'use_beta_false'},
    {'kwargs': {'use_gamma': False}, 'condition': 'use_gamma_false'},
    {'train': True, 'kwargs': {}},
    {'train': True,
     'kwargs': {'use_beta': False}, 'condition': 'use_beta_false'},
    {'train': True,
     'kwargs': {'use_gamma': False}, 'condition': 'use_gamma_false'},
    {'train': True,
     'kwargs': {'initial_avg_mean': 0.5}, 'condition': 'init_avg_mean'},
)
class TestBatchNormalization(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, **kwargs):
                super(Model, self).__init__()
                with self.init_scope():
                    self.bn = L.BatchNormalization(5, **kwargs)

            def __call__(self, x):
                return self.bn(x)

        self.model = Model(**self.kwargs)
        self.x = input_generator.increasing(2, 5)

    def test_output(self):
        train = getattr(self, 'train', False)
        name = 'batch_normalization'
        if not train:
            name = 'fixed_' + name
        if hasattr(self, 'condition'):
            name += '_' + self.condition

        def test_input_names(onnx_model, path):
            initializer_names = get_initializer_names(onnx_model)
            assert len(initializer_names) == 4
            assert 'param_bn_avg_mean' in initializer_names
            assert 'param_bn_avg_var' in initializer_names

        self.expect(
            self.model, self.x, name=name, train=train,
            custom_model_test_func=test_input_names)


class TestGroupNormalization(ONNXModelTest):

    def get_model(self):
        class Model(chainer.Chain):
            def __init__(self):
                super().__init__()
                with self.init_scope():
                    self.gn = L.GroupNormalization(2)

            def forward(self, x):
                return self.gn(x)
        return Model()

    def test_output(self):
        model = self.get_model()
        x = np.zeros((10, 4, 256, 256), dtype=np.float32)

        self.expect(model, x, train=True)


class TestBatchNormalizationFunction(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __call__(self, x):
                gamma = np.ones(x.shape[1:], dtype=x.dtype)
                beta = np.zeros(x.shape[1:], dtype=x.dtype)
                return F.batch_normalization(x, gamma, beta)

        self.model = Model()
        self.x = input_generator.increasing(2, 5)

    def test_output(self):

        def test_input_names(onnx_model, path):
            initializer_names = get_initializer_names(onnx_model)
            assert len(initializer_names) == 4
            assert 'BatchNormalization_0_param_avg_mean' in initializer_names
            assert 'BatchNormalization_0_param_avg_var' in initializer_names

        self.expect(
            self.model, self.x, custom_model_test_func=test_input_names)


class TestFixedBatchNormalizationFunctionImplicitInputs(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __call__(self, x):
                mean = x.array.mean(axis=0)
                var = x.array.var(axis=0)
                gamma = np.ones_like(mean, dtype=x.dtype)
                beta = np.zeros_like(mean, dtype=x.dtype)
                return F.fixed_batch_normalization(x, gamma, beta, mean, var)

        self.model = Model()
        self.x = input_generator.increasing(2, 5)

    def test_output(self):

        def test_input_names(onnx_model, path):
            initializer_names = get_initializer_names(onnx_model)
            assert len(initializer_names) == 4
            assert 'FixedBatchNormalization_0_param_avg_mean' in\
                initializer_names
            assert 'FixedBatchNormalization_0_param_avg_var' in\
                initializer_names

        self.expect(
            self.model, self.x, custom_model_test_func=test_input_names)


class TestFixedBatchNormalizationFunctionExplicitInputs(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __call__(self, x, gamma, beta, mean, var):
                return F.fixed_batch_normalization(x, gamma, beta, mean, var)

        self.model = Model()
        self.x = input_generator.increasing(2, 5)
        self.mean = self.x.mean(axis=0)
        self.var = self.x.var(axis=0)
        self.gamma = np.ones_like(self.mean, dtype=self.x.dtype)
        self.beta = np.zeros_like(self.mean, dtype=self.x.dtype)

    def test_output(self):
        self.expect(
            self.model, [self.x, self.gamma, self.beta, self.mean, self.var])
