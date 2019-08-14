import unittest
import weakref

import mock
import numpy as np
import pytest

import chainer
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr
import chainer.testing.backend
import chainerx


class TestBackward(unittest.TestCase):

    def test_no_output(self):
        chainer.backward([])
        chainer.backward([], [])

    def check_multiple_output_1arg(self, xp, skip_retain_grad_test=False):
        x = chainer.Variable(xp.array([1, 2], np.float32))
        h = x * 2
        y0 = h * 3
        y1 = h * 4
        y0.grad = xp.array([1, 10], np.float32)
        y1.grad = xp.array([100, 1000], np.float32)
        chainer.backward([y0, y1])
        testing.assert_allclose(x.grad, np.array([806, 8060], np.float32))
        if skip_retain_grad_test:
            return
        assert y0.grad is None
        assert y1.grad is None

    def check_multiple_output_2args(self, xp, skip_retain_grad_test=False):
        x = chainer.Variable(xp.array([1, 2], np.float32))
        h = x * 2
        y0 = h * 3
        y1 = h * 4
        gy0 = chainer.Variable(xp.array([1, 10], np.float32))
        gy1 = chainer.Variable(xp.array([100, 1000], np.float32))
        chainer.backward([y0, y1], [gy0, gy1])
        testing.assert_allclose(x.grad, np.array([806, 8060], np.float32))
        if skip_retain_grad_test:
            return
        assert y0.grad is None
        assert y1.grad is None

    def test_multiple_output_cpu(self):
        self.check_multiple_output_1arg(np)
        self.check_multiple_output_2args(np)

    @attr.gpu
    def test_multiple_output_gpu(self):
        self.check_multiple_output_1arg(cuda.cupy)
        self.check_multiple_output_2args(cuda.cupy)

    @attr.chainerx
    def test_multiple_output_chainerx_partially_ok(self):
        self.check_multiple_output_1arg(
            chainerx, skip_retain_grad_test=True)
        self.check_multiple_output_2args(
            chainerx, skip_retain_grad_test=True)

    # TODO(kataoka): Variable.backward with ChainerX backend unexpectedly
    # behaves like retain_grad=True
    @unittest.expectedFailure
    @attr.chainerx
    def test_multiple_output_1arg_chainerx(self):
        self.check_multiple_output_1arg(chainerx)

    # TODO(kataoka): Variable.backward with ChainerX backend unexpectedly
    # behaves like retain_grad=True
    @unittest.expectedFailure
    @attr.chainerx
    def test_multiple_output_2args_chainerx(self):
        self.check_multiple_output_2args(chainerx)

    def test_multiple_output_call_count(self):
        x = chainer.Variable(np.array([1, 2], np.float32))

        f = chainer.FunctionNode()
        f.forward = mock.MagicMock(
            side_effect=lambda xs: tuple(x * 2 for x in xs))
        f.backward = mock.MagicMock(
            side_effect=lambda _, gys: tuple(gy * 2 for gy in gys))

        h, = f.apply((x,))
        y0 = h * 3
        y1 = h * 4
        y0.grad = np.array([1, 10], np.float32)
        y1.grad = np.array([100, 1000], np.float32)
        chainer.backward([y0, y1])
        testing.assert_allclose(x.grad, np.array([806, 8060], np.float32))
        assert f.backward.call_count == 1

    def test_warn_no_grad(self):
        x = chainer.Variable(np.array(4, np.float32))
        x.grad = np.array(3, np.float32)
        y = x * 2
        with testing.assert_warns(RuntimeWarning):
            chainer.backward([y])
        testing.assert_allclose(x.grad, np.array(3, np.float32))
        assert y.grad is None

    def test_duplicate_outputs(self):
        x = chainer.Variable(np.array(0, np.float32))
        y = chainer.functions.identity(x)
        y.grad = np.array(3, np.float32)
        with testing.assert_warns(RuntimeWarning):
            chainer.backward([y, y])
        # 6 might be expected, but y.grad is used only once
        testing.assert_allclose(x.grad, np.array(3, np.float32))


# see also test_function_node.TestGradTypeCheck
class TestBackwardTypeCheck(unittest.TestCase):

    def _rand(self):
        return np.random.uniform(-1, 1, (2, 3)).astype(np.float32)

    def test_type_check(self):
        x = chainer.Variable(self._rand())
        y = x * x
        y.grad = self._rand()
        gy = chainer.Variable(self._rand())

        with self.assertRaises(TypeError):
            chainer.backward(y)
        with self.assertRaises(TypeError):
            chainer.backward([y], gy)

        chainer.backward([y])
        chainer.backward([y], [gy])


# see also test_function_node.TestGradValueCheck
class TestBackwardValueCheck(unittest.TestCase):

    def test_length_check(self):
        x = chainer.Variable(np.array(3, np.float32))
        y = chainer.functions.identity(x)
        gy = chainer.Variable(np.array(7, np.float32))

        with self.assertRaises(ValueError):
            chainer.backward([y], [])
        with self.assertRaises(ValueError):
            chainer.backward([y], [gy, gy])
        with self.assertRaises(ValueError):
            chainer.backward([], [gy])
        with self.assertRaises(ValueError):
            chainer.backward([y, y], [gy])

        chainer.backward([y], [gy])


class LoggedFunc(chainer.FunctionNode):

    def __init__(self, name, len_y, watcher):
        self.name = name
        self.len_y = len_y
        self.watcher = watcher

    def forward(self, inputs):
        self.len_x = len(inputs)
        h = sum(inputs)
        m = self.len_y
        return tuple([h.copy() for _ in range(m)])

    def backward(self, target_input_indexes, grad_outputs):
        grad_inputs = logged_func(
            'grad ' + self.name, self.len_x, self.watcher, grad_outputs)
        return grad_inputs


def logged_func(name, len_y, watcher, inputs):
    watcher.update(name)
    func = LoggedFunc(name, len_y, watcher)
    watcher.add_func(func)
    outputs = func.apply(inputs)
    names = [x.name for x in inputs]
    if len(outputs) == 1:
        y, = outputs
        y.name = '{}({})'.format(
            name, ','.join(names))
    else:
        for i, y in enumerate(outputs):
            y.name = '{}_{}({})'.format(
                name, i, ','.join(names))
    for y in outputs:
        watcher.add_variable(y)
    return outputs


class VariableWatcher(object):

    def __init__(self):
        self._seen_nodes = weakref.WeakSet()
        self._seen_funcs = weakref.WeakSet()
        self.targets = {}
        self._log = []

    def add_variable(self, v):
        assert v.node not in self._seen_nodes
        self._seen_nodes.add(v.node)
        name = v.name
        assert name is not None
        assert name not in self.targets
        self.targets[name] = weakref.ref(v)

    def add_func(self, func):
        assert func not in self._seen_funcs
        self._seen_funcs.add(func)
        name = func.name
        assert name is not None
        assert name not in self.targets
        self.targets[name] = weakref.ref(func)

    def _update(self):
        deleted = set()
        for name, ref in self.targets.items():
            obj = ref()
            if obj is None:
                deleted.add(name)
        for name in deleted:
            del self.targets[name]
        if deleted:
            self._log.append(deleted)

    def update(self, event_name):
        self._update()
        self._log.append(event_name)

    def get_log(self):
        self._update()
        log = self._log
        self._log = []
        return log


class TestDelayBackward(unittest.TestCase):

    def setUp(self):
        self.watcher = VariableWatcher()
        self._add_orig = chainer.functions.add

        def add_patched(*xs):
            self.watcher.update('+')
            y = self._add_orig(*xs)
            names = [x.name for x in xs]
            # Sort names because we don't care orders of operands of
            # additions in backward
            y.name = '({})'.format('+'.join(sorted(names)))
            self.watcher.add_variable(y)
            return y

        chainer.functions.add = add_patched

    def tearDown(self):
        chainer.functions.add = self._add_orig

    def var(self, name):
        v = chainer.Variable(np.array([2, 3], np.float32), name=name)
        self.watcher.add_variable(v)
        return v

    def func(self, name, xs, len_y):
        xs = tuple(xs)
        ys = logged_func(name, len_y, self.watcher, xs)
        ys = list(ys)
        return ys

    def test_simple_backward(self):
        x = self.var('x')
        h, = self.func('f', [x], 1)
        y, = self.func('g', [h], 1)
        del h
        y.grad_var = self.var('gy')
        backward = y.backward(return_cont=True)
        del y
        backward()
        assert x.grad is not None
        assert self.watcher.get_log() == [
            'f', 'g', {'f(x)', 'g(f(x))'},
            'grad g', {'g', 'grad g', 'gy'},
            'grad f', {'f', 'grad f', 'grad g(gy)'},
        ]

    def test_simple_backward_enable_double(self):
        x = self.var('x')
        h, = self.func('f', [x], 1)
        y, = self.func('g', [h], 1)
        del h
        y.grad_var = self.var('gy')
        backward = y.backward(return_cont=True, enable_double_backprop=True)
        del y
        backward()
        assert x.grad is not None
        assert self.watcher.get_log() == [
            'f', 'g', {'f(x)', 'g(f(x))'},
            'grad g', {'g', 'gy'},
            'grad f', {'f', 'grad g(gy)'},
        ]

        del x
        assert self.watcher.get_log() == [
            {'grad f', 'grad g', 'x', 'grad f(grad g(gy))'},
        ]

    def test_simple_double_backward(self):
        x = self.var('x')
        h, = self.func('f', [x], 1)
        y, = self.func('g', [h], 1)
        del h
        gy = self.var('gy')
        y.grad_var = gy
        backward = y.backward(return_cont=True, enable_double_backprop=True)
        del y
        backward()
        assert x.grad is not None
        assert self.watcher.get_log() == [
            'f', 'g', {'f(x)', 'g(f(x))'},
            'grad g', {'g'},
            'grad f', {'f', 'grad g(gy)'},
        ]

        x.grad_var.grad_var = self.var('ggx')
        backward = x.grad_var.backward(return_cont=True)
        del x
        backward()
        assert gy.grad is not None
        assert self.watcher.get_log() == [
            {'x', 'grad f(grad g(gy))'},
            'grad grad f',
            {'grad f', 'grad grad f', 'ggx'},
            'grad grad g',
            {'grad g', 'grad grad g', 'grad grad f(ggx)'},
        ]

    def test_simple_backward_retain(self):
        x = self.var('x')
        x1, = self.func('f', [x], 1)
        h, = self.func('g', [x1], 1)
        y, = self.func('h', [h], 1)
        del h
        y.grad_var = self.var('gy')
        backward = y.backward(return_cont=True, retain_grad=True)
        del y
        backward()
        assert x.grad is not None
        assert x1.grad is not None
        assert self.watcher.get_log() == [
            'f', 'g', 'h', {'g(f(x))', 'h(g(f(x)))'},
            'grad h',
            {'h', 'grad h', 'gy'},
            'grad g',
            {'g', 'grad g', 'grad h(gy)'},
            'grad f',
            {'grad f'},
        ]

    def test_backward_accum(self):
        x = self.var('x')
        h, = self.func('f', [x], 1)
        y0, = self.func('g0', [h], 1)
        y1, = self.func('g1', [h], 1)
        del h
        y0.grad_var = self.var('gy0')
        backward = y0.backward(return_cont=True)
        del y0
        backward()
        gx0 = x.grad
        assert gx0 is not None
        assert self.watcher.get_log() == [
            'f', 'g0', 'g1', {'f(x)', 'g0(f(x))'},
            'grad g0', {'g0', 'grad g0', 'gy0'},
            'grad f', {'grad f', 'grad g0(gy0)'},
        ]

        # h = f(x) should not be unchained
        y1.grad_var = self.var('gy1')
        backward = y1.backward(return_cont=True)
        del y1
        backward()
        assert not np.array_equal(gx0, x.grad)
        assert self.watcher.get_log() == [
            {'g1(f(x))'},
            'grad g1', {'g1', 'grad g1', 'gy1'},
            'grad f', {'grad f', 'grad g1(gy1)'},
            '+', {'f', 'grad f(grad g0(gy0))', 'grad f(grad g1(gy1))'}
        ]

    def test_complex1_backward(self):
        # resnet
        x = self.var('x')
        params = [self.var('p{}'.format(i)) for i in range(4)]
        h, = self.func('a0', self.func('f0', [x, params[0]], 1), 1)
        z, = self.func('r1', [x] + self.func('f1', [h, params[1]], 1), 1)
        del x, h
        h, = self.func('a2', self.func('f2', [z, params[2]], 1), 1)
        y, = self.func('r3', [z] + self.func('f3', [h, params[3]], 1), 1)
        del z, h
        y.grad_var = self.var('gy')
        backward = y.backward(return_cont=True)
        del y
        backward()
        for p in params:
            assert p.grad is not None
        log = self.watcher.get_log()
        h0 = 'a0(f0(x,p0))'
        z = 'r1(x,f1({h0},p1))'.format(h0=h0)
        h1 = 'a2(f2({z},p2))'.format(z=z)
        gh1 = 'grad a2(grad f3_0(grad r3_1(gy)))'
        gz = '(grad f2_0({gh1})+grad r3_0(gy))'.format(gh1=gh1)
        gh0 = 'grad a0(grad f1_0(grad r1_1({gz})))'.format(gz=gz)
        gx = '(grad f0_0({gh0})+grad r1_0({gz}))'.format(gh0=gh0, gz=gz)
        assert log == [
            'f0', 'a0', {'f0(x,p0)'}, 'f1', 'r1', {'x', h0, 'f1('+h0+',p1)'},
            'f2', 'a2', {'f2('+z+',p2)'},
            'f3', 'r3', {z, h1, 'f3('+h1+',p3)', 'r3('+z+',f3('+h1+',p3))'},
            'grad r3', {'r3', 'grad r3', 'gy'},
            'grad f3', {'f3', 'grad f3', 'grad r3_1(gy)'},
            'grad a2', {'a2', 'grad a2', 'grad f3_0(grad r3_1(gy))'},
            'grad f2', {'grad f2', gh1},
            '+', {'f2', 'grad r3_0(gy)', 'grad f2_0('+gh1+')'},
            'grad r1', {'r1', 'grad r1', gz},
            'grad f1', {'f1', 'grad f1', 'grad r1_1('+gz+')'},
            'grad a0', {'a0', 'grad a0', 'grad f1_0(grad r1_1('+gz+'))'},
            'grad f0', {'grad f0', gh0},
            '+', {'f0', 'grad r1_0('+gz+')', 'grad f0_0('+gh0+')', gx},
        ]

    def test_raise_continuation_twice(self):
        x = self.var('x')
        y, = self.func('f', [x], 1)
        y.grad_var = self.var('gy')
        backward = y.backward(return_cont=True)
        del y
        backward()
        with pytest.raises(RuntimeError):
            backward()

    def test_continuation_twice_backward_compat(self):
        x = self.var('x')
        y, = self.func('f', [x], 1)
        y.grad_var = self.var('gy')
        backward = y.backward
        backward()
        y.grad_var = self.var('gy2')
        backward()

    def check_backward_return_none(self, del_y):
        # Variable.backward should return None because it will find the error
        # `cont = y.backward(); cont()`, because it raises
        # TypeError: 'NoneType' object is not callable
        x = self.var('x')
        y, = self.func('f', [x], 1)
        y.grad_var = self.var('gy')

        ret = y.backward()
        assert ret is None

        backward = y.backward(return_cont=True)
        del y
        ret = backward()
        assert ret is None


testing.run_module(__name__, __file__)
