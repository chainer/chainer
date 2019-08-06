import copy
import unittest
import warnings

import mock
import numpy as np
import pytest

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import optimizer
from chainer import optimizers
from chainer import serializer
from chainer import testing
from chainer.testing import attr
import chainerx

if chainerx.is_available():
    import chainerx.testing


_backend_params = [
    # NumPy
    {},
    {'use_ideep': 'always'},
    # CuPy
    {'use_cuda': True, 'cuda_device': 0},
    {'use_cuda': True, 'cuda_device': 1},
    # ChainerX
    {'use_chainerx': True, 'chainerx_device': 'native:0'},
    {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
]


class TestHyperparameter(unittest.TestCase):

    def setUp(self):
        self.parent = optimizer.Hyperparameter()
        self.parent.x = 1
        self.parent.y = 2
        self.child = optimizer.Hyperparameter(self.parent)
        self.child.y = 3
        self.child.z = 4

    def test_getattr(self):
        self.assertTrue(hasattr(self.parent, 'x'))
        self.assertEqual(self.parent.x, 1)
        self.assertTrue(hasattr(self.parent, 'y'))
        self.assertEqual(self.parent.y, 2)
        self.assertFalse(hasattr(self.parent, 'z'))

        self.assertTrue(hasattr(self.child, 'x'))
        self.assertEqual(self.child.x, 1)
        self.assertTrue(hasattr(self.child, 'y'))
        self.assertEqual(self.child.y, 3)
        self.assertTrue(hasattr(self.child, 'z'))
        self.assertEqual(self.child.z, 4)

    def test_get_dict(self):
        self.assertEqual(self.parent.get_dict(), {'x': 1, 'y': 2})
        self.assertEqual(self.child.get_dict(), {'x': 1, 'y': 3, 'z': 4})

    def test_repr(self):
        self.assertEqual(repr(self.parent), 'Hyperparameter(x=1, y=2)')
        self.assertEqual(repr(self.child), 'Hyperparameter(x=1, y=3, z=4)')

    def test_deep_copy(self):
        parent_copy, child_copy = copy.deepcopy([self.parent, self.child])
        self.assertEqual(self.child.get_dict(), child_copy.get_dict())
        self.assertEqual(self.parent.get_dict(), parent_copy.get_dict())
        self.assertIs(child_copy.parent, parent_copy)


class DummyDeserializer(serializer.Deserializer):

    def __init__(self, target):
        super(DummyDeserializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        if value is None:
            value = self.target[key]
        elif isinstance(value, np.ndarray):
            np.copyto(value, self.target[key])
        else:
            value = type(value)(np.asarray(self.target[key]))
        return value


def _create_update_rule(has_states):
    class SimpleUpdateRule(optimizer.UpdateRule):
        def update_core_cpu(self, param):
            pass

        def update_core_gpu(self, param):
            pass

    def _init_state(data):
        state = update_rule.state
        state['a'] = 0
        state['b'] = np.array([1, 2, 3], dtype=np.float32)

    update_rule = SimpleUpdateRule()
    update_rule.update_core_cpu = mock.MagicMock(
        wraps=update_rule.update_core_cpu)
    update_rule.update_core_gpu = mock.MagicMock(
        wraps=update_rule.update_core_gpu)
    update_rule.update_core_chainerx = mock.MagicMock(
        wraps=update_rule.update_core_chainerx)
    if has_states:
        update_rule.init_state = _init_state
    return update_rule


def _create_var():
    data = np.ones((2, 3), np.float32)
    grad = np.ones_like(data)
    var = chainer.Variable(data, grad=grad)
    return var


@testing.backend.inject_backend_tests(
    [
        'test_update',
        'test_add_hook',
        'test_add_hook_with_name',
        'test_add_hook_with_function_name',
    ],
    _backend_params)
class TestUpdateRule(unittest.TestCase):

    def setUp(self):
        self.update_rule = _create_update_rule(has_states=False)
        self.var = _create_var()

    def check_update(self, backend_config):
        var = self.var
        var.to_device(backend_config.device)
        update_rule = self.update_rule

        update_rule.update(var)

        xp = backend_config.xp

        # First check update_core_chainerx.
        # If xp is chainerx, fallback xp is assigned to it for the second
        # check.
        if xp is chainerx:
            self.assertEqual(
                self.update_rule.update_core_chainerx.call_count, 1)
            xp = backend_config.device.fallback_device.xp
        else:
            self.assertEqual(
                self.update_rule.update_core_chainerx.call_count, 0)

        # Secondly check update_core_cpu and _gpu.
        if xp is np:
            self.assertEqual(update_rule.update_core_cpu.call_count, 1)
            self.assertEqual(update_rule.update_core_gpu.call_count, 0)
        elif xp is cuda.cupy:
            self.assertEqual(self.update_rule.update_core_cpu.call_count, 0)
            self.assertEqual(self.update_rule.update_core_gpu.call_count, 1)

    def test_update(self, backend_config):
        self.check_update(backend_config)

    def test_add_hook(self, backend_config):
        hook = mock.MagicMock()
        self.update_rule.add_hook(hook)

        self.check_update(backend_config)

        self.assertEqual(hook.call_count, 1)
        args = hook.call_args_list[0][0]
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], self.update_rule)
        self.assertIs(args[1], self.var)

    def test_add_hook_with_name(self, backend_config):
        hook = mock.MagicMock()
        self.update_rule.add_hook(hook, name='hook')

        self.check_update(backend_config)

        self.assertEqual(hook.call_count, 1)
        args = hook.call_args_list[0][0]
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], self.update_rule)
        self.assertIs(args[1], self.var)

    def test_remove_hook(self, backend_config):
        hook = mock.MagicMock()
        self.update_rule.add_hook(hook, name='hook')
        self.update_rule.remove_hook('hook')

        self.check_update(backend_config)

        self.assertEqual(hook.call_count, 0)

    def test_add_hook_with_function_name(self, backend_config):
        hook_body = mock.MagicMock()

        def foo(update_rule, data, grad):
            hook_body(update_rule, data, grad)

        self.update_rule.add_hook(foo)
        self.update_rule.remove_hook('foo')

        self.check_update(backend_config)

        self.assertEqual(hook_body.call_count, 0)

    def test_add_hook_no_name(self):
        class CallableWithoutName(object):
            def __call__(self, update_rule, param):
                pass

        with self.assertRaises(ValueError):
            self.update_rule.add_hook(CallableWithoutName())

    def test_add_hook_duplicated_name(self):
        self.update_rule.add_hook(mock.MagicMock(), name='foo')
        with self.assertRaises(KeyError):
            self.update_rule.add_hook(mock.MagicMock(), name='foo')

    def test_remove_hook_not_exist(self):
        with self.assertRaises(KeyError):
            self.update_rule.remove_hook('foo')

    def test_disabled_update_rule(self):
        self.update_rule.update_core = mock.MagicMock()
        self.update_rule.enabled = False
        self.update_rule.update(self.var)
        self.assertEqual(self.update_rule.update_core.call_count, 0)

        self.update_rule.enabled = True
        self.update_rule.update(self.var)
        self.assertEqual(self.update_rule.update_core.call_count, 1)


@testing.backend.inject_backend_tests(None, _backend_params)
class TestOptimizerSerialize(unittest.TestCase):

    def setUp(self):
        self.update_rule = _create_update_rule(has_states=True)

    def get_target(self, backend_config):
        target = {}
        target['t'] = 100
        target['a'] = 1
        target['b'] = (
            backend_config.get_array(np.array([2, 3, 4], dtype=np.float32)))
        return target

    def test_deserialize(self, backend_config):
        target = self.get_target(backend_config)
        self.update_rule.serialize(DummyDeserializer(target))

        self.assertEqual(self.update_rule.t, target['t'])
        self.assertIsNotNone(self.update_rule.state)
        self.assertEqual(self.update_rule.state['a'], target['a'])
        backend_config.xp.testing.assert_array_equal(
            self.update_rule.state['b'], target['b'])

    def test_deserialize_by_strict_deserializer(self, backend_config):
        target = self.get_target(backend_config)
        del target['a']
        with self.assertRaises(KeyError):
            self.update_rule.serialize(DummyDeserializer(target))

    def test_deserialize_by_nonstrict_deserializer(self, backend_config):
        target = self.get_target(backend_config)
        target['a'] = None
        self.update_rule.serialize(DummyDeserializer(target))

        self.assertEqual(self.update_rule.t, target['t'])
        self.assertIsNone(self.update_rule.state)

    def test_deserialize_disabled_update_rule_by_strict_deserializer(
            self, backend_config):
        self.update_rule.enabled = False
        target = self.get_target(backend_config)
        del target['a']
        self.update_rule.serialize(DummyDeserializer(target))

        self.assertEqual(self.update_rule.t, target['t'])
        self.assertIsNone(self.update_rule.state)


@testing.backend.inject_backend_tests(None, _backend_params)
@testing.backend.inject_backend_tests(None, _backend_params)
class TestUpdateRuleCopyState(unittest.TestCase):

    def setUp(self):
        self.update_rule = _create_update_rule(has_states=True)

    def test_state_copy(self, backend_config, _):
        def update_core(param):
            self.assertIsInstance(self.update_rule.state['a'], int)
            self.assertTrue(
                backend_config.device.is_array_supported(
                    self.update_rule.state['b']))

        self.update_rule.update_core = update_core
        var = _create_var()
        var.to_device(backend_config.device)
        self.update_rule.update(var)

    def test_state_copy_to_another_device(
            self, backend_config1, backend_config2):
        def update_core(param):
            self.assertIsInstance(self.update_rule.state['a'], int)
            self.assertTrue(
                backend_config2.device.is_array_supported(
                    self.update_rule.state['b']))

        var1 = _create_var()
        var1.to_device(backend_config1.device)
        # call update with arrays on GPU 0 (tested by another method)
        self.update_rule.update_core = lambda param: None
        self.update_rule.update(var1)
        # check if it copies the states correctly when arrays on another device
        # are passed
        self.update_rule.update_core = update_core
        var2 = _create_var()
        var2.to_device(backend_config2.device)
        self.update_rule.update(var2)


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.optimizer = optimizer.Optimizer()

    def test_new_epoch(self):
        self.optimizer.new_epoch()
        self.assertEqual(1, self.optimizer.epoch)

    def test_invalid_new_epoch(self):
        self.optimizer.use_auto_new_epoch = True
        with self.assertRaises(RuntimeError):
            self.optimizer.new_epoch()

    def test_auto_new_epoch(self):
        self.optimizer.use_auto_new_epoch = True
        self.optimizer.new_epoch(auto=True)
        self.assertEqual(1, self.optimizer.epoch)

    def test_invalid_auto_new_epoch(self):
        with self.assertRaises(RuntimeError):
            self.optimizer.new_epoch(auto=True)


@attr.chainerx
class TestOptimizerWithChainerxImplementation(unittest.TestCase):
    # This test ensures an optimizer can update ChainerX array by overriding
    # update_core_chainerx().

    def test_upate(self):
        initial_p = np.array([1., 2., 3.], np.float32)
        x = chainerx.array([2., 4., 6.], np.float32)

        expected_p = 4. * initial_p - 6. * backend.CpuDevice().send(x)

        class ChainerxUpdateRule(optimizer.UpdateRule):
            call_count = 0

            def update_core_chainerx(self, param):
                # p <= 3 * p - 2 * (dy/dp)
                array = param.array
                t1 = param.array.as_grad_stopped() * 3.
                t2 = param.grad.as_grad_stopped() * 2.
                delta = t1 - t2
                array += delta
                self.call_count += 1

        class ChainerxOptimizer(optimizer.GradientMethod):
            def create_update_rule(self):
                return ChainerxUpdateRule(self.hyperparam)

        class Link(chainer.Link):
            def __init__(self):
                super(Link, self).__init__()
                with self.init_scope():
                    self.p = chainer.Parameter(initial_p)

            def forward(self, x):
                return 3. * x * self.p

        link = Link()
        link.to_device('native:0')
        y = link(x)
        y.backward()
        optimizer_ = ChainerxOptimizer()
        optimizer_.setup(link)
        optimizer_.update()

        assert link.p.update_rule.call_count == 1
        np.testing.assert_array_equal(
            backend.CpuDevice().send(link.p.array), expected_p)


class TestOptimizerHook(unittest.TestCase):

    def setUp(self):
        self.optimizer = optimizer.Optimizer()
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3))

    def test_add_hook(self):
        h1 = mock.MagicMock(timing='pre')
        h1.call_for_each_param = False
        self.optimizer.setup(self.target)
        self.optimizer.add_hook(h1, 'h1')
        self.optimizer.call_hooks()
        h1.assert_called_with(self.optimizer)

    def test_add_hook_call_for_each_param(self):
        h1 = mock.MagicMock(timing='pre')
        h1.call_for_each_param = True
        self.optimizer.setup(self.target)
        self.optimizer.add_hook(h1, 'h1')
        self.optimizer.call_hooks()
        h1.assert_called_with(self.target.param.update_rule, self.target.param)

    def test_remove_hook(self):
        h1 = mock.MagicMock(timing='pre')
        self.optimizer.setup(self.target)
        self.optimizer.add_hook(h1, 'h1')
        self.optimizer.remove_hook('h1')
        self.optimizer.call_hooks()
        self.assertFalse(h1.called)

    def test_duplicated_hook(self):
        self.optimizer.setup(self.target)
        self.optimizer.add_hook(lambda s: None, 'h1', timing='pre')
        with self.assertRaises(KeyError):
            self.optimizer.add_hook(lambda s: None, 'h1', timing='pre')

    def test_invalid_hook(self):
        self.optimizer.setup(self.target)
        with self.assertRaises(TypeError):
            self.optimizer.add_hook(1)

    def test_add_hook_before_setup(self):
        with self.assertRaises(RuntimeError):
            self.optimizer.add_hook(lambda s: None, 'h1')


class SimpleLink(chainer.Link):

    def __init__(self, w, g):
        super(SimpleLink, self).__init__()
        with self.init_scope():
            self.param = chainer.Parameter(w)
            self.param.grad = g


@testing.backend.inject_backend_tests(['test_update'], _backend_params)
class TestGradientMethod(unittest.TestCase):

    def setUp(self):
        self.optimizer = chainer.GradientMethod()
        self.target = chainer.ChainList(
            SimpleLink(np.arange(3).astype(np.float32),
                       np.arange(3).astype(np.float32)),
            SimpleLink(np.arange(3).astype(np.float32),
                       np.arange(3).astype(np.float32)))
        self.optimizer.create_update_rule = mock.MagicMock

    def test_setup(self):
        create_update_rule = mock.MagicMock()
        target = self.target
        optimizer = self.optimizer
        optimizer.create_update_rule = create_update_rule
        optimizer.setup(target)

        self.assertEqual(create_update_rule.call_count, 2)
        self.assertEqual(create_update_rule.call_args_list[0], [(), {}])
        self.assertEqual(create_update_rule.call_args_list[1], [(), {}])

    def test_update(self, backend_config):
        target = self.target
        optimizer = self.optimizer
        target.to_device(backend_config.device)
        optimizer.setup(target)

        self.assertEqual(optimizer.t, 0)

        optimizer.update()

        self.assertEqual(optimizer.t, 1)

        param1 = target[0].param
        param2 = target[1].param
        param1.update_rule.update.assert_called_once_with(param1)
        param2.update_rule.update.assert_called_once_with(param2)


@testing.backend.inject_backend_tests(None, _backend_params)
@testing.parameterize(*testing.product({
    'override_pattern': [
        'generic',  # only update_core() is overridden
        'cpu_gpu',  # update_core_{cpu,gpu} are overridden
        'cpu_gpu_chx',  # update_core_{cpu,gpu,chainerx} are overridden
    ],
}))
class TestGradientMethodUpdate(unittest.TestCase):
    """Ensures UpdateRule's appropriate methods are called, for various
    override patterns and parameters with various conditions."""

    def create(self, device):

        class MyLink(chainer.Link):
            def __init__(self):
                super(MyLink, self).__init__()
                with self.init_scope():
                    self.p1 = chainer.Parameter()  # uninitialized

                    self.p2 = chainer.Parameter(  # initialized, with grad
                        np.array([3, 2], np.float32))
                    self.p2.grad = np.array([13, 12], np.float32)

                    self.p3 = chainer.Parameter(  # initialized, without grad
                        np.array([5, 7], np.float32))

        call_record = []
        override_pattern = self.override_pattern

        class MyUpdateRule(optimizer.UpdateRule):
            if override_pattern == 'generic':
                def update_core(self, param):
                    call_record.append(('update_core', param))

            elif override_pattern == 'cpu_gpu':
                def update_core_cpu(self, param):
                    call_record.append(('update_core_cpu', param))

                def update_core_gpu(self, param):
                    call_record.append(('update_core_gpu', param))

            elif override_pattern == 'cpu_gpu_chx':
                def update_core_cpu(self, param):
                    call_record.append(('update_core_cpu', param))

                def update_core_gpu(self, param):
                    call_record.append(('update_core_gpu', param))

                def update_core_chainerx(self, param):
                    call_record.append(('update_core_chainerx', param))

            else:
                assert False, override_pattern

        class MyOptimizer(optimizer.GradientMethod):
            def create_update_rule(self):
                return MyUpdateRule()

        optimizer_ = MyOptimizer()
        target = MyLink()
        target.to_device(device)
        optimizer_.setup(target)

        return optimizer_, call_record

    def test_update(self, backend_config):
        device = backend_config.device
        override_pattern = self.override_pattern
        optimizer, call_record = self.create(device)

        optimizer.update()

        self.assertEqual(len(call_record), 3)

        # Detemine the expected method name that was called.
        if override_pattern == 'generic':
            method_name = 'update_core'
        elif override_pattern == 'cpu_gpu':
            if isinstance(device, backend.ChainerxDevice):
                xp = device.fallback_device.xp
            else:
                xp = device.xp

            if xp is np:
                method_name = 'update_core_cpu'
            else:
                assert xp is cuda.cupy
                method_name = 'update_core_gpu'
        elif override_pattern == 'cpu_gpu_chx':
            if isinstance(device, backend.ChainerxDevice):
                method_name = 'update_core_chainerx'
            elif device.xp is np:
                method_name = 'update_core_cpu'
            else:
                assert device.xp is cuda.cupy
                method_name = 'update_core_gpu'
        else:
            assert False, override_pattern

        # Check call record.
        # TODO(niboshi): Check the param argument as well.
        self.assertEqual(call_record[0][0], method_name)
        self.assertEqual(call_record[1][0], method_name)
        self.assertEqual(call_record[2][0], method_name)


@testing.backend.inject_backend_tests(None, _backend_params)
@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2)],
    'dtype': [np.float16, np.float32, np.float64],
    'loss_scale': [None, 1, 10],
}))
class TestGradientMethodLossScale(unittest.TestCase):

    def setUp(self):
        param0_data = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        param0_grad = np.copy(param0_data)
        param1_data = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        param1_grad = np.copy(param1_data)
        self.target = chainer.ChainList(
            SimpleLink(param0_data, param0_grad),
            SimpleLink(param1_data, param1_grad))
        lr = 1.0
        if self.loss_scale is not None:
            lr = self.loss_scale
            for i in range(2):
                self.target[i].param._loss_scale = self.loss_scale
        # TODO(niboshi): Do not use SGD in GradientMethod test
        self.optimizer = chainer.optimizers.SGD(lr)

    def test_update(self, backend_config):
        if backend_config.device.name == '@cupy:1':
            # TODO(niboshi): Fix it
            raise unittest.SkipTest(
                'Loss scale does not work with cupy multi-device.')
        target = self.target
        optimizer = self.optimizer
        target.to_device(backend_config.device)
        optimizer.setup(target)
        optimizer.update()
        xp = backend.get_array_module(target[0].param)
        expected_data = xp.zeros(self.shape, dtype=self.dtype)
        rtol, atol = 1e-4, 1e-5
        if self.dtype is np.float16:
            rtol, atol = 1e-1, 1e-2
        for i in range(2):
            testing.assert_allclose(
                target[i].param.data, expected_data,
                rtol=rtol, atol=atol)


@testing.backend.inject_backend_tests(None, _backend_params)
class TestCleargradHook(unittest.TestCase):

    def setUp(self):
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3))

    def test_cleargrad(self, backend_config):

        class CleargradHook(object):

            name = 'Cleargrad'
            timing = 'pre'

            def __init__(self, _):
                pass

            def __call__(self, opt):
                for param in opt.target.params():
                    # Clear all grads
                    param.cleargrad()

        target = self.target
        target.to_device(backend_config.device)
        # TODO(niboshi): Do not use SGD in GradientMethod test
        opt = optimizers.SGD(lr=1)
        opt.setup(target)
        opt.add_hook(CleargradHook(self))
        opt.add_hook(DummyHook(self))

        opt.update()


class DummyOptimizer(chainer.GradientMethod):

    def __init__(self, test):
        super(DummyOptimizer, self).__init__()
        self.test = test

    def create_update_rule(self):
        return mock.MagicMock()


class DummyHook(object):

    name = 'Dummy'
    timing = 'pre'

    def __init__(self, test):
        self.test = test

    def __call__(self, opt):
        for param in opt.target.params():
            # Confirm all grads are not None
            self.test.assertIsNotNone(param.grad)


@testing.backend.inject_backend_tests(None, _backend_params)
class TestGradientMethodClearGrads(unittest.TestCase):

    def setUp(self):
        self.optimizer = DummyOptimizer(self)
        self.target = SimpleLink(
            np.arange(3).astype(np.float32),
            np.arange(3).astype(np.float32))
        self.optimizer.setup(self.target)
        self.optimizer.add_hook(DummyHook(self))

    def test_update(self, backend_config):
        target = self.target
        optimizer = self.optimizer
        target.to_device(backend_config.device)
        target.cleargrads()
        optimizer.update()


class TestDeprecatedOptimizerHooksEmitsWarning(unittest.TestCase):

    def setUp(self):
        self.context = warnings.catch_warnings(record=True)
        self.warnings = self.context.__enter__()
        warnings.filterwarnings(action='always', category=DeprecationWarning)

    def tearDown(self):
        self.context.__exit__()

    def test_gradient_clipping(self):
        chainer.optimizer.GradientClipping(1.)
        self.assertEqual(len(self.warnings), 1)
        self.assertIs(self.warnings[-1].category, DeprecationWarning)

    def test_gradient_hard_clipping(self):
        chainer.optimizer.GradientHardClipping(1., 2.)
        self.assertEqual(len(self.warnings), 1)
        self.assertIs(self.warnings[-1].category, DeprecationWarning)

    def test_gradient_noise(self):
        chainer.optimizer.GradientNoise(1.)
        self.assertEqual(len(self.warnings), 1)
        self.assertIs(self.warnings[-1].category, DeprecationWarning)

    def test_lasso(self):
        chainer.optimizer.Lasso(1.)
        self.assertEqual(len(self.warnings), 1)
        self.assertIs(self.warnings[-1].category, DeprecationWarning)

    def test_weight_decay(self):
        chainer.optimizer.WeightDecay(1.)
        self.assertEqual(len(self.warnings), 1)
        self.assertIs(self.warnings[-1].category, DeprecationWarning)


@testing.parameterize(*testing.product({
    # None: dtype is not given by initializer.
    # Otherwise: it's given by initializer.
    'dtype': [None, np.float16, np.float32, np.float64]
}))
class TestUpdateRuleUseFp32Update(unittest.TestCase):

    def test_uninitialized_parameter(self):
        dtype = self.dtype

        def initializer(array):
            assert False  # never called

        # Set initializer.dtype to specify the parameter's dtype
        if dtype is not None:
            initializer.dtype = dtype

        # Create an uninitialized parameter
        param = chainer.Parameter(initializer)
        assert param.array is None
        if dtype is not None:
            assert param.dtype == dtype

        # Create an update rule with custom update_core
        record = []
        update_rule = chainer.UpdateRule()

        def update_core(param):
            # param.dtype may not be retrieved because it can be uninitialized
            # and dtype is not given (i.e. self.dtype is None)
            try:
                param_dtype = param.dtype
            except RuntimeError:
                param_dtype = None
            record.append({
                'param': param,
                'dtype': param_dtype,
            })

        update_rule.update_core = update_core

        # Enable fp32 update
        update_rule.use_fp32_update()
        # Call update_rule.update
        update_rule.update(param)

        if dtype == np.float16:
            assert record[0]['param'] is not param
            assert record[0]['dtype'] == np.float32
        else:
            assert record[0]['param'] is param
            assert record[0]['dtype'] == dtype

        # The original parameter is kept uninitialized and its dtype is
        # unchanged.
        assert param.array is None
        if dtype is not None:
            assert param.dtype == dtype
        else:
            with pytest.raises(RuntimeError):
                param.dtype


testing.run_module(__name__, __file__)
