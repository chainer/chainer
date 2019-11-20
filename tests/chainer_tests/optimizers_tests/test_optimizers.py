import pickle
import unittest

import numpy as np
import six

import chainer
from chainer import functions as F
from chainer import optimizers
from chainer import testing


_all_optimizers = [
    'AdaDelta',
    'AdaGrad',
    'Adam',
    'AdamW',
    'AMSGrad',
    'AdaBound',
    'AMSBound',
    'CorrectedMomentumSGD',
    'MomentumSGD',
    'MSVAG',
    'NesterovAG',
    'RMSprop',
    'RMSpropGraves',
    'SGD',
    'SMORMS3',
]


_parameterize_optimizers = testing.parameterize(*testing.product({
    'optimizer_impl': [getattr(chainer.optimizers, o) for o in _all_optimizers]
}))


class SimpleChain(chainer.Chain):

    def __init__(self, shape=()):
        super(SimpleChain, self).__init__()
        w_np = np.asarray(np.random.randn(*shape)).astype(np.float32)
        with self.init_scope():
            self.w = chainer.Parameter(w_np, name='w')

    def __call__(self, x):
        return F.sum((x - self.w) ** 2)


class TestAllOptimizersCoverage(unittest.TestCase):
    # Checks _all_optimizers covers all the built-in optimizers.

    def test_all_optimizers_coverage(self):
        module = chainer.optimizers
        module_optimizers = []
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and issubclass(obj, chainer.Optimizer)):
                module_optimizers.append(name)

        assert sorted(_all_optimizers) == sorted(module_optimizers)


@testing.backend.inject_backend_tests(
    None,
    [
        # CPU
        {},
        # Intel
        {'use_ideep': True},
        # CUDA
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ]
)
@testing.parameterize(*(
    # Optimizers constructed with default arguments
    [
        {
            'optimizer': o,
            'kwargs': {}
        }
        for o in _all_optimizers]
    # https://chainer/chainer/issues/7424
    + [
        {
            'optimizer': 'Adam',
            'kwargs': {'weight_decay_rate': 0.5},
        }]
))
@testing.parameterize(*testing.product(
    {'shape': [(2, 3), (), (1, 0, 2)]}
))
class TestOptimizer(unittest.TestCase):

    def test_optimizer(self, backend_config):
        device = backend_config.device
        target = SimpleChain(self.shape)
        target.to_device(device)
        optimizer_cls = getattr(chainer.optimizers, self.optimizer)
        optimizer = optimizer_cls(**self.kwargs)
        optimizer.setup(target)

        x_np = np.asarray(np.random.randn(*self.shape)).astype(np.float32)
        x = chainer.Variable(device.send(x_np))

        # Just ensures no error occurs. No numerical check is performed.
        optimizer.update(target, x)


@_parameterize_optimizers
class TestOptimizerHyperparameter(unittest.TestCase):

    def setUp(self):
        self.target = chainer.Link()
        with self.target.init_scope():
            self.target.w = chainer.Parameter()

    def create(self, *args, **kwargs):
        self.optimizer = self.optimizer_impl(*args, **kwargs)
        self.optimizer.setup(self.target)

    def get_hyperparam(self, name):
        return getattr(self.target.w.update_rule.hyperparam, name)

    def test_hyperparams(self):
        # TODO(niboshi): The following optimizers do not pass this test
        # because their __init__ do not accept some hyperparameters.
        # The test should be fixed.
        if self.optimizer_impl in (
                chainer.optimizers.AdamW,
                chainer.optimizers.AMSGrad,
                chainer.optimizers.AdaBound,
                chainer.optimizers.AMSBound,
        ):
            raise unittest.SkipTest(
                'The optimizer is incompatible with this test')

        self.create()
        default = self.optimizer.hyperparam.get_dict()
        for name, default_value in six.iteritems(default):
            self.create()
            self.assertEqual(self.get_hyperparam(name), default_value)
            new_value = default_value + 0.1
            self.create(**{name: new_value})
            self.assertEqual(self.get_hyperparam(name), new_value)


class WeightSaveHook(object):
    name = 'WeightSaveHook'
    call_for_each_param = True

    def __init__(self):
        self.value = None

    def __call__(self, rule, param):
        p, g = param.data, param.grad
        if p is None or g is None:
            return
        self.value = np.copy(p)


@_parameterize_optimizers
class TestOptimizerHooks(unittest.TestCase):

    def setUp(self):
        self.target = SimpleChain()

    def create(self, *args, **kwargs):
        self.optimizer = self.optimizer_impl(*args, **kwargs)
        self.optimizer.setup(self.target)

    def get_hyperparam(self, name):
        return getattr(self.target.w.update_rule.hyperparam, name)

    def test_hooks(self):
        w_pre = np.copy(self.target.w.data)
        h_pre = WeightSaveHook()
        h_post = WeightSaveHook()
        self.create()
        self.optimizer.add_hook(h_pre, timing='pre')
        self.optimizer.add_hook(h_post, name='WeightSaveHookPost',
                                timing='post')

        x = chainer.Variable(np.array(5., dtype=np.float32))
        self.optimizer.update(self.target, x)
        w_post = np.copy(self.target.w.data)

        self.assertEqual(w_pre, h_pre.value)
        self.assertEqual(w_post, h_post.value)
        self.assertNotEqual(h_pre.value, h_post.value)

    def test_hooks_auto(self):
        w_pre = np.copy(self.target.w.data)
        h_pre = WeightSaveHook()
        h_pre.timing = 'pre'
        h_post = WeightSaveHook()
        h_post.timing = 'post'
        self.create()
        self.optimizer.add_hook(h_pre, timing='auto')
        self.optimizer.add_hook(h_post, name='WeightSaveHookPost',
                                timing='auto')

        x = chainer.Variable(np.array(5., dtype=np.float32))
        self.optimizer.update(self.target, x)
        w_post = np.copy(self.target.w.data)

        self.assertEqual(w_pre, h_pre.value)
        self.assertEqual(w_post, h_post.value)
        self.assertNotEqual(h_pre.value, h_post.value)


@_parameterize_optimizers
class TestOptimizerPickable(unittest.TestCase):

    def setUp(self):
        self.target = SimpleChain()

    def create(self, *args, **kwargs):
        self.optimizer = self.optimizer_impl(*args, **kwargs)
        self.optimizer.setup(self.target)

    def get_hyperparam(self, name):
        return getattr(self.target.w.update_rule.hyperparam, name)

    def test_new_pickle(self):
        self.create()
        pickled_opt = pickle.dumps(self.optimizer)

        x = chainer.Variable(np.array(5., dtype=np.float32))
        self.optimizer.update(self.target, x)
        w_post = np.copy(self.target.w.data)
        # Pickle has saved a copy of the target
        opt = pickle.loads(pickled_opt)
        opt.update(opt.target, x)
        pickled_w_post = np.copy(opt.target.w.data)

        self.assertEqual(w_post, pickled_w_post)

    def test_updated_pickle(self):
        self.create()

        x = chainer.Variable(np.array(5., dtype=np.float32))
        self.optimizer.update(self.target, x)
        pickled_opt = pickle.dumps(self.optimizer)

        self.optimizer.update(self.target, x)
        w_post = np.copy(self.target.w.data)
        # Pickle has saved a copy of the target
        opt = pickle.loads(pickled_opt)
        opt.update(opt.target, x)
        pickled_w_post = np.copy(opt.target.w.data)

        self.assertEqual(w_post, pickled_w_post)


@_parameterize_optimizers
class TestOptimizerLossScaling(unittest.TestCase):

    def setUp(self):
        self.target = SimpleChain()

    def create(self, *args, **kwargs):
        self.optimizer = self.optimizer_impl(*args, **kwargs)
        self.optimizer.setup(self.target)

    def test_invalid_configs(self):
        self.create()
        with self.assertRaises(ValueError):
            self.optimizer.loss_scaling(interval=0)
        with self.assertRaises(ValueError):
            self.optimizer.loss_scaling(scale=-1)


@testing.backend.inject_backend_tests(
    None,
    [
        # CPU
        {},
        # Intel
        {'use_ideep': True},
        # CUDA
        {'use_cuda': True, 'cuda_device': 0},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ]
)
class TestAdamW(unittest.TestCase):

    def test_adam_w(self, backend_config):
        xp = backend_config.xp
        device = backend_config.device

        link = chainer.Link(x=(1,))
        link.to_device(device)

        opt = optimizers.Adam(eta=0.5, weight_decay_rate=0.1)
        opt.setup(link)

        link.x.data.fill(1)
        link.x.grad = device.send(xp.ones_like(link.x.data))

        opt.update()

        # compare against the value computed with v5 impl
        testing.assert_allclose(link.x.data, np.array([0.9495]),
                                atol=1e-7, rtol=1e-7)


@testing.backend.inject_backend_tests(
    None,
    [
        # CPU
        {},
        # Intel
        {'use_ideep': True},
        # CUDA
        {'use_cuda': True, 'cuda_device': 0},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ]
)
class TestAMSGrad(unittest.TestCase):

    def test_amsgrad(self, backend_config):
        device = backend_config.device

        link = chainer.Link(x=(4,))
        x = link.x
        x.data.fill(0)
        link.to_device(device)

        opt = optimizers.Adam(alpha=0.01, beta2=0.7, amsgrad=True)
        opt.setup(link)

        x.grad = device.send(np.array([1, -1, 10, -10], np.float32))
        opt.update()
        testing.assert_allclose(
            x.update_rule.state['v'],
            [0.3, 0.3, 30, 30],
            atol=1e-7, rtol=1e-7)
        testing.assert_allclose(
            x.data,
            [-0.01, 0.01, -0.01, 0.01],
            atol=1e-7, rtol=1e-7)

        x.grad = device.send(np.array([-10, -10, -1, -1], np.float32))
        opt.update()
        testing.assert_allclose(
            x.update_rule.state['v'],
            [30.21, 30.21, 21.3, 21.3],
            atol=1e-7, rtol=1e-7)
        testing.assert_allclose(
            x.update_rule.state['vhat'],
            [30.21, 30.21, 30, 30],
            atol=1e-7, rtol=1e-7)
        testing.assert_allclose(
            x.data,
            # result with NumPy
            [-0.00377703, 0.01745388, -0.01548985, 0.01686232],
            atol=1e-7, rtol=1e-7)


testing.run_module(__name__, __file__)
