import copy
import unittest
import warnings

import mock
import numpy
import pytest

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import initializers
from chainer import testing
from chainer.testing import attr
import chainerx


def _assert_variable_array_equal(var, expected_array):
    assert var.shape == expected_array.shape
    assert var.dtype == expected_array.dtype
    _assert_arrays_equal(var.data, expected_array)


def _assert_arrays_equal(array, expected_array):
    array = backend.CpuDevice().send(array)
    assert array.shape == expected_array.shape
    assert array.dtype == expected_array.dtype
    assert (array == expected_array).all()


class LinkTestBase(object):

    def setUp(self):
        x_shape_0 = 2
        x_shape_1 = numpy.int64(3)
        self.link = chainer.Link(x=((x_shape_0, x_shape_1), 'd'),
                                 u=(None, 'd'))
        with self.link.init_scope():
            self.link.y = chainer.Parameter(shape=(2,))
            self.link.v = chainer.Parameter()
        self.p = numpy.array([1, 2, 3], dtype='f')
        self.link.add_persistent('p', self.p)
        self.link.name = 'a'
        self.link.x.update_rule = chainer.UpdateRule()
        self.link.x.update_rule.enabled = False
        self.link.u.update_rule = chainer.UpdateRule()
        if cuda.available:
            self.current_device_id = cuda.cupy.cuda.get_device_id()

    def tearDown(self):
        if cuda.available \
                and cuda.cupy.cuda.get_device_id() != self.current_device_id:
            cuda.Device(self.current_device_id).use()

    def check_param_init(self, name, shape, dtype, data_value=numpy.nan):
        self.assertTrue(hasattr(self.link, name))
        var = getattr(self.link, name)
        self.assertEqual(var.name, name)
        self.assertIsInstance(var, chainer.Parameter)
        self.assertEqual(var.data.shape, shape)
        self.assertEqual(var.data.dtype, dtype)
        numpy.testing.assert_array_equal(
            backend.CpuDevice().send(var.data), data_value)
        self.assertEqual(var.grad.shape, shape)
        self.assertEqual(var.grad.dtype, dtype)
        numpy.testing.assert_array_equal(
            backend.CpuDevice().send(var.grad), numpy.nan)

    def check_param_uninit(self, name, initializer=None):
        self.assertTrue(hasattr(self.link, name))
        var = getattr(self.link, name)
        self.assertIsInstance(var, chainer.Parameter)
        self.assertEqual(var.name, name)
        self.assertIsNone(var.data)
        if initializer is not None:
            self.assertIs(var.initializer, initializer)


class TestLink(LinkTestBase, unittest.TestCase):

    def test_init(self):
        self.check_param_init('x', (2, 3), 'd')
        self.check_param_init('y', (2,), 'f')
        self.check_param_uninit('u')
        self.link.u.initialize((2, 3))
        self.check_param_init('u', (2, 3), 'd')
        self.check_param_uninit('v')
        self.link.v.initialize((2, 3))
        self.check_param_init('v', (2, 3), 'f')

    def test_assign_param_outside_of_init_scope(self):
        p = chainer.Parameter()
        self.link.p = p
        self.assertTrue(all(p is not param for param in self.link.params()))

    def test_assign_var_in_init_scope(self):
        p = chainer.Variable()
        with self.link.init_scope():
            self.link.p = p
        self.assertTrue(all(p is not param for param in self.link.params()))

    def test_call_injected_with_mixin(self):
        call = mock.MagicMock()
        call.return_value = 3

        class CallMixin(object):
            __call__ = call

        class InjectedLink(chainer.Link, CallMixin):
            pass

        link = InjectedLink()
        ret = link(1, a=2)

        call.assert_called_once_with(1, a=2)
        assert ret == call.return_value

    def test_add_param(self):
        self.link.add_param('z', (2, 3))
        self.check_param_init('z', (2, 3), 'f')

        self.link.add_param('w', (2, 3), dtype='d')
        self.check_param_init('w', (2, 3), 'd')

        self.link.add_param('r')
        self.check_param_uninit('r')
        self.link.r.initialize((2, 3))
        self.check_param_init('r', (2, 3), 'f')

        self.link.add_param('s', dtype='d')
        self.check_param_uninit('s')
        self.link.s.initialize((2, 3))
        self.check_param_init('s', (2, 3), 'd')

        initializer = initializers.Zero('d')
        self.link.add_param('t', initializer=initializer)
        self.check_param_uninit('t', initializer)
        self.link.t.initialize((2, 3))
        self.check_param_init('t', (2, 3), 'd', 0)

    def test_add_param_direct_initialization(self):
        z = numpy.random.rand(2, 3).astype('f')
        self.link.add_param('z', initializer=z)
        self.assertIsInstance(self.link.z.data, numpy.ndarray)
        numpy.testing.assert_array_equal(self.link.z.data, z)

    def test_add_param_duplicated_with_persistent(self):
        self.link.add_persistent('z', 'abc')
        with self.assertRaises(AttributeError):
            self.link.add_param('z', (2, 3))

    def test_add_persistent(self):
        self.assertTrue(hasattr(self.link, 'p'))
        self.assertIs(self.link.p, self.p)

        self.link.add_persistent('q', 'abc')
        self.assertTrue(hasattr(self.link, 'q'))
        self.assertEqual(self.link.q, 'abc')

    def test_delete(self):
        del self.link.x
        self.assertFalse(hasattr(self.link, 'x'))
        self.assertNotIn('x', self.link._params)
        self.assertNotIn('x', self.link._persistent)

        del self.link.p
        self.assertFalse(hasattr(self.link, 'p'))
        self.assertNotIn('p', self.link._params)
        self.assertNotIn('p', self.link._persistent)

    def test_copy_with_share_mode(self):
        link = self.link.copy(mode='share')
        self.assertIsInstance(link._params, set)
        self.assertIsInstance(link._persistent, set)
        self.assertTrue(hasattr(link, 'x'))
        self.assertTrue(hasattr(link, 'y'))
        self.assertTrue(hasattr(link, 'u'))
        self.assertTrue(hasattr(link, 'p'))
        self.assertIsNot(link.x, self.link.x)
        self.assertIs(link.x.array, self.link.x.array)
        self.assertIsNot(link.y, self.link.y)
        self.assertIs(link.y.array, self.link.y.array)
        self.assertIsNone(link.u.array)
        self.assertIs(link.p, self.link.p)
        self.assertIs(link.name, None)

    def test_copy_with_copy_mode(self):
        link = self.link.copy(mode='copy')
        self.assertIsInstance(link._params, set)
        self.assertIsInstance(link._persistent, set)
        self.assertTrue(hasattr(link, 'x'))
        self.assertTrue(hasattr(link, 'y'))
        self.assertTrue(hasattr(link, 'u'))
        self.assertTrue(hasattr(link, 'p'))
        self.assertIsNot(link.x, self.link.x)
        self.assertIsNot(link.x.array, self.link.x.array)
        self.assertIsNot(link.y, self.link.y)
        self.assertIsNot(link.y.array, self.link.y.array)
        self.assertIsNone(link.u.array)
        self.assertIsNot(link.p, self.link.p)
        self.assertIsNot(link.name, None)

    def test_copy_with_init_mode(self):
        self.link.u.initializer = initializers.Normal(
            dtype=self.link.u.initializer.dtype)
        self.link.u.initialize((2, 3))
        link = self.link.copy(mode='init')
        self.assertFalse(numpy.array_equal(self.link.u.array, link.u.array))
        self.assertIsInstance(link._params, set)
        self.assertIsInstance(link._persistent, set)
        self.assertTrue(hasattr(link, 'x'))
        self.assertTrue(hasattr(link, 'y'))
        self.assertTrue(hasattr(link, 'u'))
        self.assertTrue(hasattr(link, 'p'))
        self.assertIsNot(link.x, self.link.x)
        self.assertIsNot(link.x.array, self.link.x.array)
        self.assertIsNot(link.y, self.link.y)
        self.assertIsNot(link.y.array, self.link.y.array)
        self.assertIsNot(link.p, self.link.p)
        self.assertIsNot(link.name, None)

    @attr.gpu
    def test_copy_and_to_gpu_init(self):
        cupy = cuda.cupy
        l0 = self.link
        l1 = l0.copy()
        self.assertIs(l0.x.data, l1.x.data)
        l1.to_gpu()
        self.assertIsNot(l0.x.data, l1.x.data)
        self.assertIsInstance(l0.x.data, numpy.ndarray)
        self.assertIsInstance(l1.x.data, cupy.ndarray)

    @attr.gpu
    def test_copy_and_to_gpu_uninit(self):
        cupy = cuda.cupy
        l0 = self.link
        l1 = l0.copy()
        self.assertIs(l0.device.xp, numpy)
        self.assertIsNone(l0.u.data)
        self.assertIsNone(l1.u.data)
        l1.to_gpu()
        self.assertIs(l0.device.xp, numpy)
        self.assertIsNone(l0.u.data)
        l1.u.initialize((2, 3))
        self.assertIsNone(l0.u.data)
        self.assertIsInstance(l1.u.data, cupy.ndarray)

    @attr.multi_gpu(2)
    def test_copy_and_to_gpu_uninit_multi_gpu(self):
        cupy = cuda.cupy
        l0 = self.link
        l1 = l0.copy()
        l2 = l0.copy()
        self.assertIsNone(l0.u.data)
        self.assertIsNone(l1.u.data)
        self.assertIsNone(l2.u.data)
        l1.to_gpu()
        l1.u.initialize((2, 3))
        l2.to_gpu()
        l2.u.initialize((2, 3))
        self.assertIsNone(l0.u.data)
        self.assertIsInstance(l1.u.data, cupy.ndarray)
        self.assertIsInstance(l2.u.data, cupy.ndarray)
        self.assertNotEqual(l1.u.data.data, l2.u.data.data)

    def _check_deepcopy(self, link):
        self.assertIsInstance(link._params, set)
        self.assertIsInstance(link._persistent, set)
        self.assertTrue(hasattr(link, 'x'))
        self.assertTrue(hasattr(link, 'y'))
        self.assertTrue(hasattr(link, 'u'))
        self.assertTrue(hasattr(link, 'p'))
        self.assertIsNot(link.x, self.link.x)
        self.assertIsNot(link.x.data, self.link.x.data)
        numpy.testing.assert_array_equal(cuda.to_cpu(link.x.data),
                                         cuda.to_cpu(self.link.x.data))
        self.assertIsNot(link.y, self.link.y)
        self.assertIsNot(link.y.data, self.link.y.data)
        numpy.testing.assert_array_equal(cuda.to_cpu(link.y.data),
                                         cuda.to_cpu(self.link.y.data))
        self.assertIsNone(link.u.data)
        self.assertIsNot(link.p, self.link.p)
        self.assertEqual(link.name, self.link.name)

    def test_deepcopy(self):
        link = copy.deepcopy(self.link)
        self._check_deepcopy(link)
        self.assertEqual(link.device.xp, numpy)

    @attr.multi_gpu(2)
    def test_deepcopy_multi_device(self):
        device_id = 1
        self.link.to_gpu(device_id)
        link = copy.deepcopy(self.link)
        self._check_deepcopy(link)
        self.assertEqual(link.device.device, cuda.Device(device_id))
        self.assertEqual(link.x.data.device.id, device_id)
        self.assertEqual(link.y.data.device.id, device_id)

    def test_to_cpu_on_cpu(self):
        x = self.link.x.data
        gx = self.link.x.grad
        y = self.link.y.data
        gy = self.link.y.grad
        p = self.link.p
        self.link.to_cpu()
        self.assertIs(self.link.x.data, x)
        self.assertIs(self.link.x.grad, gx)
        self.assertIs(self.link.y.data, y)
        self.assertIs(self.link.y.grad, gy)
        self.assertIsNone(self.link.u.data)
        self.assertIsNone(self.link.u.grad)
        self.assertIs(self.link.p, p)

    @attr.gpu
    def test_to_cpu(self):
        self.link.to_gpu()
        self.link.to_cpu()
        self.link.v.initialize((2, 3))
        self.assertIs(self.link.xp, numpy)
        self.assertIsInstance(self.link.x.data, numpy.ndarray)
        self.assertIsInstance(self.link.x.grad, numpy.ndarray)
        self.assertIsInstance(self.link.y.data, numpy.ndarray)
        self.assertIsInstance(self.link.y.grad, numpy.ndarray)
        self.assertIsNone(self.link.u.data)
        self.assertIsNone(self.link.u.grad)
        self.assertIsInstance(self.link.v.data, numpy.ndarray)
        self.assertIsInstance(self.link.v.grad, numpy.ndarray)
        self.assertIsInstance(self.link.p, numpy.ndarray)

    @attr.gpu
    def test_to_gpu(self):
        cupy = cuda.cupy
        self.link.to_gpu()
        self.link.v.initialize((2, 3))
        self.assertIs(self.link.xp, cupy)
        self.assertIsInstance(self.link.x.data, cupy.ndarray)
        self.assertIsInstance(self.link.x.grad, cupy.ndarray)
        self.assertIsInstance(self.link.y.data, cupy.ndarray)
        self.assertIsInstance(self.link.y.grad, cupy.ndarray)
        self.assertIsNone(self.link.u.data)
        self.assertIsNone(self.link.u.grad)
        self.assertIsInstance(self.link.v.data, cupy.ndarray)
        self.assertIsInstance(self.link.v.grad, cupy.ndarray)
        self.assertIsInstance(self.link.p, cupy.ndarray)

    @attr.multi_gpu(2)
    def test_to_gpu_different_current_device(self):
        cuda.Device(1).use()
        self.link.to_gpu(0)
        self.assertEqual(self.link.device.device, cuda.Device(0))

    @attr.multi_gpu(2)
    def test_to_gpu_different_device(self):
        self.link.to_gpu(0)
        self.assertEqual(self.link.device.device, cuda.Device(0))
        self.assertEqual(self.link.x.data.device, cuda.Device(0))
        self.assertEqual(self.link.x.grad.device, cuda.Device(0))
        self.assertEqual(self.link.y.data.device, cuda.Device(0))
        self.assertEqual(self.link.y.grad.device, cuda.Device(0))
        self.assertEqual(self.link.p.device, cuda.Device(0))
        self.link.to_gpu(1)
        self.assertEqual(self.link.device.device, cuda.Device(1))
        self.assertEqual(self.link.x.data.device, cuda.Device(0))
        self.assertEqual(self.link.x.grad.device, cuda.Device(0))
        self.assertEqual(self.link.y.data.device, cuda.Device(0))
        self.assertEqual(self.link.y.grad.device, cuda.Device(0))
        self.assertEqual(self.link.p.device, cuda.Device(0))

    @attr.multi_gpu(2)
    def test_to_gpu_current_device(self):
        cuda.Device(1).use()
        self.link.to_gpu()
        self.assertEqual(self.link.device.device, cuda.Device(1))

    def test_params(self):
        params = list(self.link.params())
        self.assertEqual([id(p) for p in params],
                         [id(self.link.u), id(self.link.v),
                          id(self.link.x), id(self.link.y)])

    def test_params_skip_uninit(self):
        params = list(self.link.params(include_uninit=False))
        self.assertEqual([id(p) for p in params],
                         [id(self.link.x), id(self.link.y)])

    def test_namedparams(self):
        namedparams = list(self.link.namedparams())
        self.assertEqual([(name, id(p)) for name, p in namedparams],
                         [('/u', id(self.link.u)), ('/v', id(self.link.v)),
                          ('/x', id(self.link.x)), ('/y', id(self.link.y))])

    def test_namedparams_skip_uninit(self):
        namedparams = list(self.link.namedparams(include_uninit=False))
        self.assertEqual([(name, id(p)) for name, p in namedparams],
                         [('/x', id(self.link.x)), ('/y', id(self.link.y))])

    def test_links(self):
        links = list(self.link.links())
        self.assertIs(links[0], self.link)

    def test_links_skipself(self):
        links = list(self.link.links(skipself=True))
        self.assertFalse(links)  # empty

    def test_namedlinks(self):
        pl = list(self.link.namedlinks())
        self.assertEqual(len(pl), 1)
        self.assertEqual(pl[0][0], '/')
        self.assertIs(pl[0][1], self.link)

    def _setup_test_copyparams(self):
        self.link.x.grad.fill(0)
        self.link.y.grad.fill(1)
        self.link.u.initialize((2, 3))
        self.link.u.data.fill(0)
        self.link.u.grad.fill(1)
        self.link.v.cleargrad()
        gx = self.link.x.grad.copy()
        gy = self.link.y.grad.copy()
        gu = self.link.u.grad.copy()

        l = chainer.Link()
        with l.init_scope():
            l.x = chainer.Parameter(shape=(2, 3))
            l.y = chainer.Parameter(shape=2)
            l.u = chainer.Parameter(shape=(2, 3))
            l.v = chainer.Parameter(shape=(3, 2))
        l.x.data.fill(2)
        l.x.grad.fill(3)
        l.y.data.fill(4)
        l.y.grad.fill(5)
        l.u.data.fill(6)
        l.u.grad.fill(7)
        l.v.data.fill(8)
        l.v.grad.fill(9)
        l.add_persistent('p', numpy.full_like(self.link.p, 10))

        return l, (gx, gy, gu)

    def _check_copyparams(self, l, gs):
        gx, gy, gu = gs
        numpy.testing.assert_array_equal(self.link.x.data, l.x.data)
        numpy.testing.assert_array_equal(self.link.x.grad, gx)
        numpy.testing.assert_array_equal(self.link.y.data, l.y.data)
        numpy.testing.assert_array_equal(self.link.y.grad, gy)
        numpy.testing.assert_array_equal(self.link.u.data, l.u.data)
        numpy.testing.assert_array_equal(self.link.u.grad, gu)
        numpy.testing.assert_array_equal(self.link.v.data, l.v.data)
        numpy.testing.assert_array_equal(self.link.v.grad, None)

    def test_copyparams(self):
        l, gs = self._setup_test_copyparams()
        self.link.copyparams(l)
        self._check_copyparams(l, gs)
        numpy.testing.assert_array_equal(self.link.p, l.p)

    def test_copyparams_no_copy_persistent(self):
        orig_p = self.link.p.copy()

        l, gs = self._setup_test_copyparams()
        numpy.testing.assert_array_equal(False, orig_p == l.p)
        self.link.copyparams(l, copy_persistent=False)

        self._check_copyparams(l, gs)
        numpy.testing.assert_array_equal(self.link.p, orig_p)

    def test_cleargrads(self):
        self.link.cleargrads()
        self.assertIsNone(self.link.x.grad)
        self.assertIsNone(self.link.y.grad)
        self.link.u.initialize((2, 3))
        self.link.v.initialize((2, 3))
        self.assertIsNone(self.link.u.grad)
        self.assertIsNone(self.link.v.grad)

    def test_zerograds(self):
        gx_expect = numpy.zeros_like(self.link.x.data)
        gy_expect = numpy.zeros_like(self.link.y.data)
        with testing.assert_warns(DeprecationWarning):
            self.link.zerograds()
        numpy.testing.assert_array_equal(self.link.x.grad, gx_expect)
        numpy.testing.assert_array_equal(self.link.y.grad, gy_expect)
        self.link.u.initialize((2, 3))
        self.link.v.initialize((2, 3))
        gu_expect = numpy.zeros_like(self.link.u.data)
        gv_expect = numpy.zeros_like(self.link.v.data)
        numpy.testing.assert_array_equal(self.link.u.grad, gu_expect)
        numpy.testing.assert_array_equal(self.link.v.grad, gv_expect)

    def test_addgrads(self):
        l = chainer.Link()
        with l.init_scope():
            l.x = chainer.Parameter(shape=(2, 3),
                                    initializer=initializers.NaN('d'))
            l.y = chainer.Parameter(shape=2)
            l.u = chainer.Parameter(shape=(2, 3))
            l.v = chainer.Parameter()
        l.x.grad.fill(1)
        l.y.grad.fill(2)
        l.u.grad.fill(3)

        self.link.x.grad.fill(-1)
        self.link.y.grad.fill(-2)
        self.link.u.cleargrad()

        self.link.addgrads(l)

        gx_expect = numpy.zeros_like(l.x.grad)
        gy_expect = numpy.zeros_like(l.y.grad)
        gu_expect = l.u.grad
        numpy.testing.assert_array_equal(self.link.x.grad, gx_expect)
        numpy.testing.assert_array_equal(self.link.y.grad, gy_expect)
        numpy.testing.assert_array_equal(self.link.u.grad, gu_expect)
        self.assertIsNone(self.link.v.grad, None)

    def test_serialize(self):
        serializer = mock.MagicMock(return_value=3)
        l = chainer.Link()
        with l.init_scope():
            l.x = chainer.Parameter(shape=(2, 3))
            l.y = chainer.Parameter(shape=2)
        l.add_persistent('z', 1)
        l.serialize(serializer)
        self.assertEqual(serializer.call_count, 3)
        serializer.assert_any_call('x', l.x.data)
        serializer.assert_any_call('y', l.y.data)
        serializer.assert_any_call('z', 1)
        self.assertEqual(l.z, 3)

    def test_serialize_param_shape_placeholder(self):
        serializer = mock.MagicMock(return_value=3)
        l = chainer.Link()
        with l.init_scope():
            l.y = chainer.Parameter(shape=2)
            l.x = chainer.Parameter()
        l.x.initialize((2, 3))
        l.add_persistent('z', 1)
        l.serialize(serializer)
        self.assertEqual(serializer.call_count, 3)
        serializer.assert_any_call('x', l.x.data)
        serializer.assert_any_call('y', l.y.data)
        serializer.assert_any_call('z', 1)
        self.assertEqual(l.z, 3)

    def test_serialize_deserialize_to_uninitialized_param(self):
        ret = numpy.random.rand(2, 3).astype('f')
        serializer = mock.MagicMock(return_value=ret)
        l = chainer.Link()
        with l.init_scope():
            l.x = chainer.Parameter()
        l.serialize(serializer)
        self.assertEqual(serializer.call_count, 1)
        serializer.assert_any_call('x', None)
        self.assertIsInstance(l.x.data, numpy.ndarray)
        numpy.testing.assert_array_equal(l.x.data, ret)

    def test_enable_update(self):
        self.link.enable_update()
        self.assertTrue(self.link.x.update_rule.enabled)
        self.assertTrue(self.link.u.update_rule.enabled)

    def test_disable_update(self):
        self.link.disable_update()
        self.assertFalse(self.link.x.update_rule.enabled)
        self.assertFalse(self.link.u.update_rule.enabled)

    def test_update_enabled(self):
        self.assertTrue(self.link.update_enabled)
        self.link.disable_update()
        self.assertFalse(self.link.update_enabled)
        self.link.enable_update()
        self.assertTrue(self.link.update_enabled)

    def test_count_params(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            assert self.link.count_params() == 8
        assert len(w) == 2
        assert w[0].category is UserWarning

        self.link.u.initialize((2, 3))
        self.link.v.initialize((2, 3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.link.count_params()
        assert not w


@testing.backend.inject_backend_tests(
    None,
    [
        # NumPy
        {},
        # CuPy
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
@attr.chainerx
class TestLinkFromToChainerx(LinkTestBase, unittest.TestCase):

    def test_from_chainerx(self, backend_config):
        self.link.to_device(backend_config.device)
        self.link.from_chainerx()

        source_device = backend_config.device

        self.check_param_init('x', (2, 3), 'd')
        self.check_param_init('y', (2,), 'f')
        self.check_param_uninit('u')

        if source_device.xp is chainerx:
            backend_name = source_device.device.backend.name
            if backend_name == 'native':
                expected_device = backend.CpuDevice()
            elif backend_name == 'cuda':
                expected_device = backend.GpuDevice.from_device_id(
                    source_device.device.index)
        else:
            expected_device = source_device

        self.assertEqual(self.link._device, expected_device)

    def test_to_chainerx(self, backend_config):
        self.link.to_device(backend_config.device)
        self.link.to_chainerx()

        source_device = backend_config.device

        self.check_param_init('x', (2, 3), 'd')
        self.check_param_init('y', (2,), 'f')
        self.check_param_uninit('u')

        if source_device.xp is chainerx:
            expected_device = source_device
        elif source_device.xp is numpy:
            expected_device = backend.ChainerxDevice(
                chainerx.get_device('native', 0))
        elif source_device.xp is cuda.cupy:
            expected_device = backend.ChainerxDevice(
                chainerx.get_device('cuda', source_device.device.id))
        else:
            assert False

        self.assertEqual(self.link._device, expected_device)


class TestLinkMissingInitCall(unittest.TestCase):
    # Tests for detecting incorrectly written Link subclasses in which
    # the call to Link.__init__ is missing

    expected_message = r'^Link\.__init__\(\) has not been called\.$'

    def test_missing1(self):
        # Nothing is done in __init__.
        # The fault should be detected no later than __call__().

        class Derived(chainer.Link):
            def __init__(self):
                pass

            def forward(self, x):
                return x

        with pytest.raises(RuntimeError, match=self.expected_message):
            link = Derived()
            link(numpy.array([1, 2], numpy.float32))

    def test_missing2(self):
        # init_scope is called.
        # The fault should be detected at init_scope.

        class Derived(chainer.Link):
            def __init__(self):
                with self.init_scope():
                    pass

        with pytest.raises(RuntimeError, match=self.expected_message):
            Derived()

    def test_missing3(self):
        # add_param is called.
        # The fault should be detected at add_param.

        class Derived(chainer.Link):
            def __init__(self):
                self.add_param('p1', (2, 3), numpy.float32)

        with pytest.raises(RuntimeError, match=self.expected_message):
            Derived()


class TestLinkRepeat(unittest.TestCase):

    def setUp(self):

        class Layer(chainer.Link):
            def __init__(self):
                super(Layer, self).__init__()
                with self.init_scope():
                    self.x = chainer.Parameter(
                        chainer.initializers.Normal(), shape=(2, 3))

            def forward(self):
                pass

        self.link = Layer()

    def test_no_repeat(self):
        ret = self.link.repeat(0)
        self.assertEqual(len(ret), 0)

    def test_repeat_with_init(self):
        ret = self.link.repeat(2, mode='init')
        self.assertEqual(len(ret), 2)
        # Both should be different objects from the original link
        self.assertIsNot(ret[0], self.link)
        self.assertIsNot(ret[1], self.link)
        # Object IDs of elements should be different
        self.assertIsNot(ret[0], ret[1])
        self.assertIsNot(ret[0].x, ret[1].x)
        # But shape and type of paratmeres shuld be same
        self.assertEqual(ret[0].x.shape, self.link.x.shape)
        self.assertEqual(ret[0].x.dtype, self.link.x.dtype)
        self.assertEqual(ret[0].x.shape, ret[1].x.shape)
        self.assertEqual(ret[0].x.dtype, ret[1].x.dtype)
        # Parameters are re-initialized, so the values should be different
        self.assertFalse(numpy.all(ret[0].x.array == ret[1].x.array))

    def test_repeat_with_copy(self):
        ret = self.link.repeat(2, mode='copy')
        self.assertEqual(len(ret), 2)
        # Both should be different objects from the original link
        self.assertIsNot(ret[0], self.link)
        self.assertIsNot(ret[1], self.link)
        # Object IDs of elements should be different
        self.assertIsNot(ret[0], ret[1])
        self.assertIsNot(ret[0].x, ret[1].x)
        # But shape, type, and value of paratmeres shuld be same
        self.assertEqual(ret[0].x.shape, self.link.x.shape)
        self.assertEqual(ret[0].x.dtype, self.link.x.dtype)
        self.assertEqual(ret[0].x.shape, ret[1].x.shape)
        self.assertEqual(ret[0].x.dtype, ret[1].x.dtype)
        numpy.testing.assert_array_equal(ret[0].x.array, ret[1].x.array)

    def test_repeat_with_share(self):
        ret = self.link.repeat(2, mode='share')
        self.assertEqual(len(ret), 2)
        # Both should be different objects from the original link
        self.assertIsNot(ret[0], self.link)
        self.assertIsNot(ret[1], self.link)
        # Object IDs of elements should be different
        self.assertIsNot(ret[0], ret[1])
        self.assertIsNot(ret[0].x, ret[1].x)
        # But the array objects should be the same
        self.assertIs(ret[0].x.array, ret[1].x.array)
        # But shape, type, and value of paratmeres shuld be same
        self.assertEqual(ret[0].x.shape, self.link.x.shape)
        self.assertEqual(ret[0].x.dtype, self.link.x.dtype)
        self.assertEqual(ret[0].x.shape, ret[1].x.shape)
        self.assertEqual(ret[0].x.dtype, ret[1].x.dtype)
        numpy.testing.assert_array_equal(ret[0].x.array, ret[1].x.array)


class CountParameter(chainer.Parameter):

    def __init__(self, v):
        super(CountParameter, self).__init__(v.data, name=v.name)
        self.data = v.data
        self.grad = v.grad
        self.count_to_cpu = 0
        self.count_to_gpu = 0
        self.count_to_device = 0
        self.count_zerograd = 0

    def to_cpu(self):
        self.count_to_cpu += 1
        return super(CountParameter, self).to_cpu()

    def to_gpu(self, device=None):
        self.count_to_gpu += 1
        return super(CountParameter, self).to_gpu(device)

    def to_device(self, device=None):
        self.count_to_device += 1
        return super(CountParameter, self).to_device(device)

    def zerograd(self):
        self.count_zerograd += 1
        super(CountParameter, self).zerograd()


class ChainTestBase(object):

    def setUp(self):
        # Schematic:
        # c2
        # - c1
        #   - l1 (x: uninitialized with shape=(2, 3))
        #   - l2 (x: uninitialized with shape=2)
        # - l3   (x: uninitialized without shape)

        self.l1 = chainer.Link()
        with self.l1.init_scope():
            self.l1.x = chainer.Parameter(shape=(2, 3))

        self.l2 = chainer.Link()
        with self.l2.init_scope():
            self.l2.x = chainer.Parameter(shape=2)

        self.l3 = chainer.Link()
        with self.l3.init_scope():
            self.l3.x = chainer.Parameter()

        self.c1 = chainer.Chain()
        with self.c1.init_scope():
            self.c1.l1 = self.l1
        self.c1.add_link('l2', self.l2)

        self.c2 = chainer.Chain()
        with self.c2.init_scope():
            self.c2.c1 = self.c1
            self.c2.l3 = self.l3

    def set_count_parameters(self):
        self.l1.x = CountParameter(self.l1.x)
        self.l2.x = CountParameter(self.l2.x)
        self.l3.x = CountParameter(self.l3.x)


class TestChain(ChainTestBase, unittest.TestCase):

    def test_init(self):
        self.assertIs(self.c1.l1, self.l1)
        self.assertIs(self.c1['l1'], self.l1)
        self.assertEqual(self.l1.name, 'l1')

        self.assertIs(self.c2.c1, self.c1)
        self.assertIs(self.c2['c1'], self.c1)
        self.assertEqual(self.c1.name, 'c1')

        self.assertIs(self.c2.l3, self.l3)
        self.assertIs(self.c2['l3'], self.l3)
        self.assertEqual(self.l3.name, 'l3')

    def test_add_link(self):
        self.assertIs(self.c1.l2, self.l2)
        self.assertEqual(self.l2.name, 'l2')

    def test_add_link_to_existing_attribute(self):
        self.l1.z = 0
        with self.assertRaises(AttributeError):
            self.l1.add_link('z', chainer.Link())

    def test_assign_link_outside_of_init_scope(self):
        l = chainer.Link()
        self.l1.l = l
        self.assertTrue(all(l is not link for link in self.l1.links()))

    def test_delete_link(self):
        del self.c1.l1
        self.assertFalse(hasattr(self.c1, 'l1'))
        self.assertNotIn('l1', self.c1._children)

    def test_copy_with_share_mode(self):
        self.l1.x.initializer = initializers.Normal(
            dtype=self.l1.x.initializer.dtype)
        self.l1.x.initialize(self.l1.x.shape)
        self.l2.x.initializer = initializers.Normal(
            dtype=self.l2.x.initializer.dtype)
        self.l2.x.initialize(self.l2.x.shape)

        c2 = self.c2.copy(mode='share')
        self.assertIs(c2.name, None)
        self.assertIsInstance(c2._children, set)
        self.assertTrue(hasattr(c2, 'c1'))
        self.assertEqual(c2.c1.name, 'c1')
        self.assertIsInstance(c2.c1._children, set)
        self.assertIsNot(c2.c1, self.c1)
        self.assertEqual(c2.c1.l1.name, 'l1')
        self.assertIsNot(c2.c1.l1, self.l1)
        self.assertIsNot(c2.c1.l1.x, self.l1.x)
        self.assertIs(c2.c1.l1.x.data, self.l1.x.data)
        self.assertIs(c2.c1.l1.x.grad, None)

        self.assertTrue(hasattr(c2.c1, 'l2'))
        self.assertEqual(c2.c1.l2.name, 'l2')
        self.assertIsNot(c2.c1.l2, self.l2)
        self.assertIsNot(c2.c1.l2.x, self.l2.x)
        self.assertIs(c2.c1.l2.x.data, self.l2.x.data)
        self.assertIs(c2.c1.l2.x.grad, None)

        self.assertTrue(hasattr(c2, 'l3'))
        self.assertEqual(c2.l3.name, 'l3')
        self.assertIsNot(c2.l3, self.l3)
        self.assertIsNot(c2.l3.x, self.l3.x)
        self.assertIs(c2.l3.x.data, self.l3.x.data)
        self.assertIs(c2.l3.x.grad, None)

    def test_copy_with_copy_mode(self):
        self.l1.x.initializer = initializers.Normal(
            dtype=self.l1.x.initializer.dtype)
        self.l1.x.initialize(self.l1.x.shape)
        self.l2.x.initializer = initializers.Normal(
            dtype=self.l2.x.initializer.dtype)
        self.l2.x.initialize(self.l2.x.shape)

        c2 = self.c2.copy(mode='copy')
        self.assertIs(c2.name, None)
        self.assertIsInstance(c2._children, set)
        self.assertTrue(hasattr(c2, 'c1'))
        self.assertEqual(c2.c1.name, 'c1')
        self.assertIsInstance(c2.c1._children, set)
        self.assertIsNot(c2.c1, self.c1)
        self.assertEqual(c2.c1.l1.name, 'l1')
        self.assertIsNot(c2.c1.l1, self.l1)
        self.assertIsNot(c2.c1.l1.x, self.l1.x)
        self.assertIsNot(c2.c1.l1.x.data, self.l1.x.data)
        self.assertTrue(numpy.array_equal(c2.c1.l1.x.data, self.l1.x.data))
        self.assertIs(c2.c1.l1.x.grad, None)

        self.assertTrue(hasattr(c2.c1, 'l2'))
        self.assertEqual(c2.c1.l2.name, 'l2')
        self.assertIsNot(c2.c1.l2, self.l2)
        self.assertIsNot(c2.c1.l2.x, self.l2.x)
        self.assertIsNot(c2.c1.l2.x.data, self.l2.x.data)
        self.assertTrue(numpy.array_equal(c2.c1.l2.x.data, self.l2.x.data))
        self.assertIs(c2.c1.l2.x.grad, None)

        self.assertTrue(hasattr(c2, 'l3'))
        self.assertEqual(c2.l3.name, 'l3')
        self.assertIsNot(c2.l3, self.l3)
        self.assertIsNot(c2.l3.x, self.l3.x)
        self.assertIs(c2.l3.x.data, self.l3.x.data)
        self.assertIs(c2.l3.x.grad, None)

    def test_copy_with_init_mode(self):
        self.l1.x.initializer = initializers.Normal(
            dtype=self.l1.x.initializer.dtype)
        self.l1.x.initialize(self.l1.x.shape)
        self.l2.x.initializer = initializers.Normal(
            dtype=self.l2.x.initializer.dtype)
        self.l2.x.initialize(self.l2.x.shape)

        c2 = self.c2.copy(mode='init')
        self.assertIs(c2.name, None)
        self.assertIsInstance(c2._children, set)
        self.assertTrue(hasattr(c2, 'c1'))
        self.assertEqual(c2.c1.name, 'c1')
        self.assertIsInstance(c2.c1._children, set)
        self.assertIsNot(c2.c1, self.c1)
        self.assertEqual(c2.c1.l1.name, 'l1')
        self.assertIsNot(c2.c1.l1, self.l1)
        self.assertIsNot(c2.c1.l1.x, self.l1.x)
        self.assertIsNot(c2.c1.l1.x.data, self.l1.x.data)
        self.assertFalse(numpy.array_equal(c2.c1.l1.x.data, self.l1.x.data))
        # _grad_initializer attribute in a copied Parameter has constant.NaN
        # after calling initilize() method
        self.assertTrue(numpy.isnan(c2.c1.l1.x.grad).all())

        self.assertTrue(hasattr(c2.c1, 'l2'))
        self.assertEqual(c2.c1.l2.name, 'l2')
        self.assertIsNot(c2.c1.l2, self.l2)
        self.assertIsNot(c2.c1.l2.x, self.l2.x)
        self.assertIsNot(c2.c1.l2.x.data, self.l2.x.data)
        self.assertFalse(numpy.array_equal(c2.c1.l2.x.data, self.l2.x.data))
        # _grad_initializer attribute in a copied Parameter has constant.NaN
        # after calling initilize() method
        self.assertTrue(numpy.isnan(c2.c1.l2.x.grad).all())

        self.assertTrue(hasattr(c2, 'l3'))
        self.assertEqual(c2.l3.name, 'l3')
        self.assertIsNot(c2.l3, self.l3)
        self.assertIsNot(c2.l3.x, self.l3.x)
        self.assertIs(c2.l3.x.data, self.l3.x.data)
        # A Parameter constructed with shape argument but not initialized
        # has None in grad
        self.assertIs(c2.l3.x.grad, None)

    def test_to_cpu_on_cpu(self):
        x1 = self.l1.x.data
        gx1 = self.l1.x.grad
        x2 = self.l2.x.data
        gx2 = self.l2.x.grad
        x3 = self.l3.x.data
        gx3 = self.l3.x.grad

        self.c2.to_cpu()
        self.assertIs(self.l1.x.data, x1)
        self.assertIs(self.l1.x.grad, gx1)
        self.assertIs(self.l2.x.data, x2)
        self.assertIs(self.l2.x.grad, gx2)
        self.assertIs(self.l3.x.data, x3)
        self.assertIs(self.l3.x.grad, gx3)

    @attr.gpu
    def test_to_cpu(self):
        self.set_count_parameters()
        self.c2.to_gpu()
        self.c2.to_cpu()
        self.assertIs(self.c2.xp, numpy)
        self.assertIs(self.c1.xp, numpy)
        self.assertIs(self.l1.xp, numpy)
        self.assertIs(self.l2.xp, numpy)
        self.assertIs(self.l3.xp, numpy)
        self.assertIsInstance(self.l1.x.data, numpy.ndarray)
        self.assertIsInstance(self.l1.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l2.x.data, numpy.ndarray)
        self.assertIsInstance(self.l2.x.grad, numpy.ndarray)
        self.assertIsNone(self.l3.x.data)
        self.assertIsNone(self.l3.x.grad)
        self.assertEqual(self.l1.x.count_to_cpu, 0)
        self.assertEqual(self.l1.x.count_to_gpu, 0)
        self.assertEqual(self.l1.x.count_to_device, 2)
        self.assertEqual(self.l2.x.count_to_cpu, 0)
        self.assertEqual(self.l2.x.count_to_gpu, 0)
        self.assertEqual(self.l2.x.count_to_device, 2)
        self.assertEqual(self.l3.x.count_to_cpu, 0)
        self.assertEqual(self.l3.x.count_to_gpu, 0)
        self.assertEqual(self.l3.x.count_to_device, 2)

        self.l3.x.initialize(3)
        self.assertIsInstance(self.l3.x.data, numpy.ndarray)
        self.assertIsInstance(self.l3.x.grad, numpy.ndarray)

    @attr.gpu
    def test_to_gpu(self):
        self.set_count_parameters()
        cupy = cuda.cupy
        self.c2.to_gpu()
        self.assertIs(self.c2.xp, cupy)
        self.assertIs(self.c1.xp, cupy)
        self.assertIs(self.l1.xp, cupy)
        self.assertIs(self.l2.xp, cupy)
        self.assertIs(self.l3.xp, cupy)
        self.assertIsInstance(self.l1.x.data, cupy.ndarray)
        self.assertIsInstance(self.l1.x.grad, cupy.ndarray)
        self.assertIsInstance(self.l2.x.data, cupy.ndarray)
        self.assertIsInstance(self.l2.x.grad, cupy.ndarray)
        self.assertIsNone(self.l3.x.data)
        self.assertIsNone(self.l3.x.grad)
        self.assertEqual(self.l1.x.count_to_gpu, 0)
        self.assertEqual(self.l1.x.count_to_device, 1)
        self.assertEqual(self.l2.x.count_to_gpu, 0)
        self.assertEqual(self.l2.x.count_to_device, 1)
        self.assertEqual(self.l3.x.count_to_gpu, 0)
        self.assertEqual(self.l3.x.count_to_device, 1)

        self.l3.x.initialize(3)
        self.assertIsInstance(self.l3.x.data, cupy.ndarray)
        self.assertIsInstance(self.l3.x.grad, cupy.ndarray)

    def test_to_device(self):
        self.set_count_parameters()
        device = backend.CpuDevice()
        self.c2.to_device(device)
        self.assertIs(self.c2.xp, numpy)
        self.assertIs(self.c1.xp, numpy)
        self.assertIs(self.l1.xp, numpy)
        self.assertIs(self.l2.xp, numpy)
        self.assertIs(self.l3.xp, numpy)
        self.assertIsInstance(self.l1.x.data, numpy.ndarray)
        self.assertIsInstance(self.l1.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l2.x.data, numpy.ndarray)
        self.assertIsInstance(self.l2.x.grad, numpy.ndarray)
        self.assertIsNone(self.l3.x.data)
        self.assertEqual(self.l1.x.count_to_device, 1)
        self.assertEqual(self.l2.x.count_to_device, 1)
        self.assertEqual(self.l3.x.count_to_device, 1)

        self.l3.x.initialize((3,))
        self.assertIsInstance(self.l3.x.data, numpy.ndarray)
        self.assertIsInstance(self.l3.x.grad, numpy.ndarray)

    def test_params(self):
        params = list(self.c2.params())
        self.assertEqual([id(p) for p in params],
                         [id(self.l1.x), id(self.l2.x), id(self.l3.x)])

    def test_params_skip_uninit(self):
        params = list(self.c2.params(include_uninit=False))
        self.assertEqual([id(p) for p in params],
                         [id(self.l1.x), id(self.l2.x)])

    def test_namedparams(self):
        namedparams = list(self.c2.namedparams())
        self.assertEqual([(name, id(p)) for name, p in namedparams],
                         [('/c1/l1/x', id(self.l1.x)),
                          ('/c1/l2/x', id(self.l2.x)),
                          ('/l3/x', id(self.l3.x))])

    def test_namedparams_skip_uninit(self):
        namedparams = list(self.c2.namedparams(include_uninit=False))
        self.assertEqual([(name, id(p)) for name, p in namedparams],
                         [('/c1/l1/x', id(self.l1.x)),
                          ('/c1/l2/x', id(self.l2.x))])

    def test_links(self):
        links = list(self.c2.links())
        self.assertEqual([id(l) for l in links],
                         [id(l) for l in [self.c2,
                                          self.c1, self.l1, self.l2,
                                          self.l3]])

    def test_links_skipself(self):
        links = list(self.c2.links(skipself=True))
        self.assertEqual([id(l) for l in links],
                         [id(l) for l in [self.c1, self.l1, self.l2,
                                          self.l3]])

    def test_namedlinks(self):
        namedlinks = list(self.c2.namedlinks())
        self.assertEqual([(name, id(l)) for name, l in namedlinks],
                         [('/', id(self.c2)),
                          ('/c1', id(self.c1)),
                          ('/c1/l1', id(self.l1)),
                          ('/c1/l2', id(self.l2)),
                          ('/l3', id(self.l3))])

    def test_namedlinks_skipself(self):
        namedlinks = list(self.c2.namedlinks(skipself=True))
        self.assertEqual([(name, id(l)) for name, l in namedlinks],
                         [('/c1', id(self.c1)),
                          ('/c1/l1', id(self.l1)),
                          ('/c1/l2', id(self.l2)),
                          ('/l3', id(self.l3))])

    def test_children(self):
        children = list(self.c2.children())
        self.assertEqual([id(c) for c in children], [id(self.c1), id(self.l3)])

    def test_copyparams(self):
        l1 = chainer.Link()
        with l1.init_scope():
            l1.x = chainer.Parameter(shape=(2, 3))
        l2 = chainer.Link()
        with l2.init_scope():
            l2.x = chainer.Parameter(shape=2)
        l3 = chainer.Link()
        with l3.init_scope():
            l3.x = chainer.Parameter(shape=3)
        c1 = chainer.Chain()
        with c1.init_scope():
            c1.l1 = l1
            c1.l2 = l2
        c2 = chainer.Chain()
        with c2.init_scope():
            c2.c1 = c1
            c2.l3 = l3
        l1.x.data.fill(0)
        l2.x.data.fill(1)
        l3.x.data.fill(2)

        self.c2.copyparams(c2)

        numpy.testing.assert_array_equal(self.l1.x.data, l1.x.data)
        numpy.testing.assert_array_equal(self.l2.x.data, l2.x.data)
        numpy.testing.assert_array_equal(self.l3.x.data, l3.x.data)

    def test_zerograds(self):
        self.set_count_parameters()
        with testing.assert_warns(DeprecationWarning):
            self.c2.zerograds()
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        self.assertEqual(self.l1.x.count_zerograd, 1)
        self.assertEqual(self.l2.x.count_zerograd, 1)
        self.assertEqual(self.l3.x.count_zerograd, 1)

        self.l3.x.initialize(3)
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.zeros(3))

    def test_addgrads(self):
        l1 = chainer.Link()
        with l1.init_scope():
            l1.x = chainer.Parameter(shape=(2, 3))
        l2 = chainer.Link()
        with l2.init_scope():
            l2.x = chainer.Parameter(shape=2)
        l3 = chainer.Link()
        with l3.init_scope():
            l3.x = chainer.Parameter(shape=3)
        c1 = chainer.Chain()
        with c1.init_scope():
            c1.l1 = l1
            c1.l2 = l2
        c2 = chainer.Chain()
        with c2.init_scope():
            c2.c1 = c1
            c2.l3 = l3
        l1.x.grad.fill(1)
        l2.x.grad.fill(2)
        l3.x.grad.fill(3)

        self.l1.x.grad.fill(-1)
        self.l2.x.grad.fill(-2)
        self.l3.cleargrads()

        self.c2.addgrads(c2)
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.full(3, 3.))

    def test_serialize(self):
        mocks = {'l1': mock.MagicMock(), 'l2': mock.MagicMock()}
        serializer = mock.MagicMock()
        serializer.__getitem__.side_effect = lambda k: mocks[k]
        self.c1.serialize(serializer)

        self.assertEqual(serializer.call_count, 0)
        self.assertEqual(serializer.__getitem__.call_count, 2)
        serializer.__getitem__.assert_any_call('l1')
        serializer.__getitem__.assert_any_call('l2')

        mocks['l1'].assert_called_with('x', self.l1.x.data)
        mocks['l2'].assert_called_with('x', self.l2.x.data)

    def test_count_params(self):
        assert self.c1.count_params() == 8

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.c2.count_params()
        assert len(w) == 1
        assert w[0].category is UserWarning

        self.c2.l3.x.initialize((3,))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.c2.count_params()
        assert not w


@testing.backend.inject_backend_tests(
    None,
    [
        # NumPy
        {},
        # CuPy
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
@attr.chainerx
class TestChainFromToChainerx(ChainTestBase, unittest.TestCase):

    def check_array_device(self, array, expected_device):
        expected_ndarray = expected_device.xp.ndarray
        self.assertIsInstance(array, expected_ndarray)
        if expected_device.xp in (chainerx, cuda.cupy):
            assert array.device == expected_device.device

    def check_expected_device(self, expected_device):
        expected_xp = expected_device.xp
        self.assertIs(self.c2.xp, expected_xp)
        self.assertIs(self.c1.xp, expected_xp)
        self.assertIs(self.l1.xp, expected_xp)
        self.assertIs(self.l2.xp, expected_xp)
        self.assertIs(self.l3.xp, expected_xp)
        self.check_array_device(self.l1.x.data, expected_device)
        self.check_array_device(self.l1.x.grad, expected_device)
        self.check_array_device(self.l2.x.data, expected_device)
        self.check_array_device(self.l2.x.grad, expected_device)
        self.assertIsNone(self.l3.x.data)

        self.l3.x.initialize((3,))
        self.check_array_device(self.l3.x.data, expected_device)
        self.check_array_device(self.l3.x.grad, expected_device)

    def test_to_chainerx(self, backend_config):
        self.set_count_parameters()
        self.c2.to_device(backend_config.device)
        self.c2.to_chainerx()

        src_device = backend_config.device
        if src_device.xp is chainerx:
            expected_device = src_device
        else:
            expected_device = (
                backend.ChainerxDevice.from_fallback_device(src_device))
        self.check_expected_device(expected_device)

    def test_from_chainerx(self, backend_config):
        self.set_count_parameters()
        self.c2.to_device(backend_config.device)
        self.c2.from_chainerx()

        src_device = backend_config.device
        if src_device.xp is chainerx:
            expected_device = src_device.fallback_device
        else:
            expected_device = src_device
        self.check_expected_device(expected_device)


class TestChainRepeat(unittest.TestCase):

    def setUp(self):
        class ChainForTest(chainer.Chain):
            def __init__(self):
                super(ChainForTest, self).__init__()
                with self.init_scope():
                    self.link = chainer.Link()

            def forward(self):
                pass

        self.chain = ChainForTest()
        self.link = self.chain.link
        with self.link.init_scope():
            self.link.x = chainer.Parameter(
                chainer.initializers.Normal(), shape=(2, 3))

    def test_no_repeat(self):
        ret = self.chain.repeat(0)
        self.assertEqual(len(ret), 0)

    def test_repeat_with_share_mode(self):
        ret = self.chain.repeat(2, mode='share')
        self.assertEqual(len(ret), 2)
        self.assertIsNot(ret[0], self.chain)
        self.assertIsNot(ret[1], self.chain)
        self.assertIsNot(ret[0], ret[1])
        self.assertIsNot(ret[0].link, self.chain.link)
        self.assertIsNot(ret[1].link, self.chain.link)
        self.assertIsNot(ret[0].link, ret[1].link)
        self.assertIsNot(ret[0].link.x, self.chain.link.x)
        self.assertIsNot(ret[1].link.x, self.chain.link.x)
        self.assertIsNot(ret[0].link.x, ret[1].link.x)
        self.assertIs(ret[0].link.x.data, self.chain.link.x.data)
        self.assertIs(ret[0].link.x.data, ret[1].link.x.data)
        self.assertEqual(ret[0].link.x.shape, self.chain.link.x.shape)
        self.assertEqual(ret[0].link.x.shape, ret[1].link.x.shape)
        self.assertEqual(ret[0].link.x.dtype, self.chain.link.x.dtype)
        self.assertEqual(ret[0].link.x.dtype, ret[1].link.x.dtype)

    def test_repeat_with_copy_mode(self):
        ret = self.chain.repeat(2, mode='copy')
        self.assertEqual(len(ret), 2)
        self.assertIsNot(ret[0], self.chain)
        self.assertIsNot(ret[1], self.chain)
        self.assertIsNot(ret[0], ret[1])
        self.assertIsNot(ret[0].link, self.chain.link)
        self.assertIsNot(ret[1].link, self.chain.link)
        self.assertIsNot(ret[0].link, ret[1].link)
        self.assertIsNot(ret[0].link.x, self.link.x)
        self.assertIsNot(ret[1].link.x, self.link.x)
        self.assertIsNot(ret[0].link.x, ret[1].link.x)
        self.assertIsNot(ret[0].link.x.data, self.chain.link.x.data)
        self.assertIsNot(ret[1].link.x.data, self.chain.link.x.data)
        self.assertIsNot(ret[0].link.x.data, ret[1].link.x.data)
        self.assertTrue(numpy.array_equal(
            ret[0].link.x.data, self.chain.link.x.data))
        self.assertTrue(numpy.array_equal(
            ret[0].link.x.data, ret[1].link.x.data))
        self.assertEqual(ret[0].link.x.shape, self.chain.link.x.shape)
        self.assertEqual(ret[0].link.x.shape, ret[1].link.x.shape)
        self.assertEqual(ret[0].link.x.dtype, self.chain.link.x.dtype)
        self.assertEqual(ret[0].link.x.dtype, ret[1].link.x.dtype)

    def test_repeat_with_init_mode(self):
        ret = self.chain.repeat(2, mode='init')
        self.assertEqual(len(ret), 2)
        self.assertIsNot(ret[0], self.chain)
        self.assertIsNot(ret[1], self.chain)
        self.assertIsNot(ret[0], ret[1])
        self.assertIsNot(ret[0].link, self.chain.link)
        self.assertIsNot(ret[1].link, self.chain.link)
        self.assertIsNot(ret[0].link.x, ret[1].link.x)
        self.assertIsNot(ret[0].link.x.data, self.chain.link.x.data)
        self.assertIsNot(ret[1].link.x.data, self.chain.link.x.data)
        self.assertIsNot(ret[0].link.x.data, ret[1].link.x.data)
        self.assertFalse(numpy.array_equal(
            ret[0].link.x.data, self.chain.link.x.data))
        self.assertFalse(numpy.array_equal(
            ret[1].link.x.data, self.chain.link.x.data))
        self.assertFalse(numpy.array_equal(
            ret[0].link.x.data, ret[1].link.x.data))
        self.assertEqual(ret[0].link.x.shape, self.chain.link.x.shape)
        self.assertEqual(ret[0].link.x.shape, ret[1].link.x.shape)
        self.assertEqual(ret[0].link.x.dtype, self.chain.link.x.dtype)
        self.assertEqual(ret[0].link.x.dtype, ret[1].link.x.dtype)


class TestChainList(unittest.TestCase):

    def setUp(self):
        self.l1 = chainer.Link()
        with self.l1.init_scope():
            self.l1.x = chainer.Parameter(shape=(2, 3))
            self.l1.y = chainer.Parameter()
        self.l2 = chainer.Link()
        with self.l2.init_scope():
            self.l2.x = chainer.Parameter(shape=2)
        self.l3 = chainer.Link()
        with self.l3.init_scope():
            self.l3.x = chainer.Parameter(shape=3)
        self.l4 = chainer.Link()
        self.l5 = chainer.Link()
        self.l6 = chainer.Link()
        self.c1 = chainer.ChainList(self.l1)
        self.c1.add_link(self.l2)
        self.c2 = chainer.ChainList(self.c1)
        self.c2.append(self.l3)
        self.c3 = chainer.ChainList(self.l4)

    def test_init(self):
        self.assertIs(self.c1[0], self.l1)
        self.assertEqual(self.l1.name, '0')
        self.assertIs(self.c2[0], self.c1)
        self.assertEqual(self.c1.name, '0')

    def test_add_link(self):
        self.assertIs(self.c1[1], self.l2)
        self.assertEqual(self.l2.name, '1')

    def test_append(self):
        self.assertIs(self.c2[1], self.l3)
        self.assertEqual(self.l3.name, '1')

    def test_setitem(self):
        self.c1[1] = self.l3
        self.assertEqual(self.l3.name, '1')

    def test_setitem_slice(self):
        self.c1.append(self.l3)  # l1 l2 l3
        self.c1[3:0:-1] = [self.l4, self.l5]  # l1 l5 l4
        self.assertEqual(len(self.c1), 3)
        self.assertEqual(self.l1.name, '0')
        self.assertEqual(self.l4.name, '2')
        self.assertEqual(self.l5.name, '1')

    def test_setitem_slice_short(self):
        self.c1.append(self.l3)  # l1 l2 l3
        self.c1[1:3] = [self.l4]  # l1 l4
        self.assertEqual(len(self.c1), 2)
        self.assertEqual(self.l1.name, '0')
        self.assertEqual(self.l4.name, '1')

    def test_setitem_slice_long(self):
        self.c1.append(self.l3)  # l1 l2 l3
        self.c1[1:3] = [self.l4, self.l5, self.l6]  # l1 l4 l5 l6
        self.assertEqual(len(self.c1), 4)
        self.assertEqual(self.l1.name, '0')
        self.assertEqual(self.l4.name, '1')
        self.assertEqual(self.l5.name, '2')
        self.assertEqual(self.l6.name, '3')

    def test_iadd(self):
        self.c2 += self.c3
        self.assertIs(len(self.c2), 3)
        self.assertEqual(self.l4.name, '2')

    def test_delete_item(self):
        del self.c2[0]
        self.assertEqual(len(self.c2), 1)
        self.assertEqual(self.l3.name, '0')

    def test_assign_param_in_init_scope(self):
        p = chainer.Parameter()
        with self.c1.init_scope():
            self.c1.p = p
        self.assertIn(p, self.c1.params())

    def test_assign_link_in_init_scope(self):
        l = chainer.Link()
        with self.c1.init_scope():
            with self.assertRaises(TypeError):
                self.c1.l = l

    def test_iter(self):
        links = list(self.c2)
        self.assertEqual(2, len(links))
        self.assertIs(links[0], self.c1)
        self.assertIs(links[1], self.l3)

    def test_len(self):
        self.assertEqual(len(self.c1), 2)
        self.assertEqual(len(self.c2), 2)

    def test_copy_with_share_mode(self):
        c2 = self.c2.copy(mode='share')
        self.l1.x.initializer = initializers.Normal(
            dtype=self.l1.x.initializer.dtype)
        self.l1.x.initialize(self.l1.x.shape)
        self.l2.x.initializer = initializers.Normal(
            dtype=self.l2.x.initializer.dtype)
        self.l2.x.initialize(self.l2.x.shape)

        self.assertIs(c2.name, None)
        self.assertIsInstance(c2._children, list)
        self.assertIsNot(c2[0], self.c1)
        self.assertEqual(c2[0].name, '0')
        self.assertIsInstance(c2[0]._children, list)
        self.assertIsNot(c2[0][0], self.l1)
        self.assertEqual(c2[0][0].name, '0')
        self.assertIsNot(c2[0][0].x, self.l1.x)
        self.assertIs(c2[0][0].x.data, self.l1.x.data)
        self.assertIs(c2[0][0].x.grad, None)

        self.assertIsNot(c2[0][1], self.l2)
        self.assertEqual(c2[0][1].name, '1')
        self.assertIsNot(c2[0][1].x, self.l2.x)
        self.assertIs(c2[0][1].x.data, self.l2.x.data)
        self.assertIs(c2[0][1].x.grad, None)

        self.assertIsNot(c2[1], self.l3)
        self.assertEqual(c2[1].name, '1')
        self.assertIsNot(c2[1].x, self.l3.x)
        self.assertIs(c2[1].x.data, self.l3.x.data)
        self.assertIs(c2[1].x.grad, None)

    def test_copy_with_copy_mode(self):
        self.l1.x.initializer = initializers.Normal(
            dtype=self.l1.x.initializer.dtype)
        self.l1.x.initialize(self.l1.x.shape)
        self.l2.x.initializer = initializers.Normal(
            dtype=self.l2.x.initializer.dtype)
        self.l2.x.initialize(self.l2.x.shape)

        c2 = self.c2.copy(mode='copy')
        self.assertIs(c2.name, None)
        self.assertIsInstance(c2._children, list)
        self.assertEqual(c2[0].name, '0')
        self.assertIsInstance(c2[0]._children, list)
        self.assertIsNot(c2[0][0], self.l1)
        self.assertEqual(c2[0][0].name, '0')
        self.assertIsNot(c2[0][0].x, self.l1.x)
        self.assertIsNot(c2[0][0].x.data, self.l1.x.data)
        self.assertTrue(numpy.array_equal(c2[0][0].x.data, self.l1.x.data))
        self.assertIs(c2[0][0].x.grad, None)

        self.assertIsNot(c2[0][1], self.l2)
        self.assertEqual(c2[0][1].name, '1')
        self.assertIsNot(c2[0][1].x, self.l2.x)
        self.assertIsNot(c2[0][1].x.data, self.l2.x.data)
        self.assertTrue(numpy.array_equal(c2[0][1].x.data, self.l2.x.data))
        self.assertIs(c2[0][1].x.grad, None)

        self.assertIsNot(c2[1], self.l3)
        self.assertEqual(c2[1].name, '1')
        self.assertIsNot(c2[1].x, self.l3.x)
        self.assertIsNot(c2[1].x.data, self.l3.x.data)
        # l3 is constructed with shape argument but not initialized
        self.assertTrue(numpy.isnan(c2[1].x.grad).all())

    def test_copy_with_init_mode(self):
        self.l1.x.initializer = initializers.Normal(
            dtype=self.l1.x.initializer.dtype)
        self.l1.x.initialize(self.l1.x.shape)
        self.l2.x.initializer = initializers.Normal(
            dtype=self.l2.x.initializer.dtype)
        self.l2.x.initialize(self.l2.x.shape)

        c2 = self.c2.copy(mode='init')
        self.assertIs(c2.name, None)
        self.assertIsInstance(c2._children, list)
        self.assertEqual(c2[0].name, '0')
        self.assertIsInstance(c2[0]._children, list)
        self.assertIsNot(c2[0][0], self.l1)
        self.assertEqual(c2[0][0].name, '0')
        self.assertIsNot(c2[0][0].x, self.l1.x)
        self.assertIsNot(c2[0][0].x.data, self.l1.x.data)
        self.assertFalse(numpy.array_equal(c2[0][0].x.data, self.l1.x.data))
        # _grad_initializer attribute in a copied Parameter has constant.NaN
        # after calling initilize() method
        self.assertTrue(numpy.isnan(c2[0][0].x.grad).all())

        self.assertIsNot(c2[0][1], self.l2)
        self.assertEqual(c2[0][1].name, '1')
        self.assertIsNot(c2[0][1].x, self.l2.x)
        self.assertIsNot(c2[0][1].x.data, self.l2.x.data)
        self.assertFalse(numpy.array_equal(c2[0][1].x.data, self.l2.x.data))
        # _grad_initializer attribute in a copied Parameter has constant.NaN
        # after calling initilize() method
        self.assertTrue(numpy.isnan(c2[0][1].x.grad).all())

        self.assertIsNot(c2[1], self.l3)
        self.assertEqual(c2[1].name, '1')
        self.assertIsNot(c2[1].x, self.l3.x)
        self.assertTrue(numpy.isnan(c2[1].x.data).all())
        self.assertTrue(numpy.isnan(c2[1].x.grad).all())

    @attr.gpu
    def test_copy_and_send_to_gpu(self):
        c2 = self.c2.copy()
        self.c2.to_gpu()
        self.assertIsInstance(self.c2[0][0].x.data, cuda.cupy.ndarray)
        self.assertIsInstance(self.c2[0][1].x.data, cuda.cupy.ndarray)
        self.assertIsInstance(c2[0][0].x.data, numpy.ndarray)
        self.assertIsInstance(c2[0][1].x.data, numpy.ndarray)

    @attr.gpu
    def test_copy_and_send_to_gpu_2(self):
        c2 = self.c2.copy()
        c2.to_gpu()
        self.assertIsInstance(self.c2[0][0].x.data, numpy.ndarray)
        self.assertIsInstance(self.c2[0][1].x.data, numpy.ndarray)
        self.assertIsInstance(c2[0][0].x.data, cuda.cupy.ndarray)
        self.assertIsInstance(c2[0][1].x.data, cuda.cupy.ndarray)

    @attr.multi_gpu(2)
    def test_copy_and_send_to_gpu_multi(self):
        c2 = self.c2.copy()
        self.c2.to_gpu(0)
        c2.to_gpu(1)
        self.assertEqual(self.c2[0][0].x.data.device.id, 0)
        self.assertEqual(self.c2[0][1].x.data.device.id, 0)
        self.assertEqual(c2[0][0].x.data.device.id, 1)
        self.assertEqual(c2[0][1].x.data.device.id, 1)

    def test_to_cpu_on_cpu(self):
        x1 = self.l1.x.data
        gx1 = self.l1.x.grad
        x2 = self.l2.x.data
        gx2 = self.l2.x.grad
        x3 = self.l3.x.data
        gx3 = self.l3.x.grad

        self.c2.to_cpu()

        self.assertIs(self.l1.x.data, x1)
        self.assertIs(self.l1.x.grad, gx1)
        self.assertIs(self.l2.x.data, x2)
        self.assertIs(self.l2.x.grad, gx2)
        self.assertIs(self.l3.x.data, x3)
        self.assertIs(self.l3.x.grad, gx3)

    @attr.gpu
    def test_to_cpu(self):
        self.c2.to_gpu()
        self.c2.to_cpu()
        self.assertIs(self.c2.xp, numpy)
        self.assertIs(self.c1.xp, numpy)
        self.assertIs(self.l1.xp, numpy)
        self.assertIs(self.l2.xp, numpy)
        self.assertIs(self.l3.xp, numpy)
        self.assertIsInstance(self.l1.x.data, numpy.ndarray)
        self.assertIsInstance(self.l1.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l2.x.data, numpy.ndarray)
        self.assertIsInstance(self.l2.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l3.x.data, numpy.ndarray)
        self.assertIsInstance(self.l3.x.grad, numpy.ndarray)

    @attr.gpu
    def test_to_gpu(self):
        cupy = cuda.cupy
        self.c2.to_gpu()
        self.assertIs(self.c2.xp, cupy)
        self.assertIs(self.c1.xp, cupy)
        self.assertIs(self.l1.xp, cupy)
        self.assertIs(self.l2.xp, cupy)
        self.assertIs(self.l3.xp, cupy)
        self.assertIsInstance(self.l1.x.data, cupy.ndarray)
        self.assertIsInstance(self.l1.x.grad, cupy.ndarray)
        self.assertIsInstance(self.l2.x.data, cupy.ndarray)
        self.assertIsInstance(self.l2.x.grad, cupy.ndarray)
        self.assertIsInstance(self.l3.x.data, cupy.ndarray)
        self.assertIsInstance(self.l3.x.grad, cupy.ndarray)

    @attr.chainerx
    def test_to_chainerx(self):
        self.c2.to_device(backend.CpuDevice())
        self.c2.to_chainerx()
        self.assertIs(self.c2.xp, chainerx)
        self.assertIs(self.c1.xp, chainerx)
        self.assertIs(self.l1.xp, chainerx)
        self.assertIs(self.l2.xp, chainerx)
        self.assertIs(self.l3.xp, chainerx)
        self.assertIsInstance(self.l1.x.data, chainerx.ndarray)
        self.assertIsInstance(self.l1.x.grad, chainerx.ndarray)
        self.assertIsInstance(self.l2.x.data, chainerx.ndarray)
        self.assertIsInstance(self.l2.x.grad, chainerx.ndarray)
        self.assertIsInstance(self.l3.x.data, chainerx.ndarray)
        self.assertIsInstance(self.l3.x.grad, chainerx.ndarray)
        expected_device = chainerx.get_device('native:0')
        self.assertIs(self.l1.x.data.device, expected_device)
        self.assertIs(self.l1.x.grad.device, expected_device)
        self.assertIs(self.l2.x.data.device, expected_device)
        self.assertIs(self.l2.x.grad.device, expected_device)
        self.assertIs(self.l3.x.data.device, expected_device)
        self.assertIs(self.l3.x.grad.device, expected_device)

    def test_to_device(self):
        device = backend.CpuDevice()
        self.c2.to_device(device)
        self.assertIs(self.c2.xp, numpy)
        self.assertIs(self.c1.xp, numpy)
        self.assertIs(self.l1.xp, numpy)
        self.assertIs(self.l2.xp, numpy)
        self.assertIs(self.l3.xp, numpy)
        self.assertIsInstance(self.l1.x.data, numpy.ndarray)
        self.assertIsInstance(self.l1.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l2.x.data, numpy.ndarray)
        self.assertIsInstance(self.l2.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l3.x.data, numpy.ndarray)
        self.assertIsInstance(self.l3.x.grad, numpy.ndarray)

    def test_params(self):
        params = list(self.c2.params())
        self.assertEqual([id(p) for p in params],
                         [id(self.l1.x), id(self.l1.y),
                          id(self.l2.x), id(self.l3.x)])

    def test_params_skip_uninit(self):
        params = list(self.c2.params(include_uninit=False))
        self.assertEqual([id(p) for p in params],
                         [id(self.l1.x), id(self.l2.x), id(self.l3.x)])

    def test_namedparams(self):
        namedparams = list(self.c2.namedparams())
        self.assertEqual([(name, id(p)) for name, p in namedparams],
                         [('/0/0/x', id(self.l1.x)),
                          ('/0/0/y', id(self.l1.y)),
                          ('/0/1/x', id(self.l2.x)),
                          ('/1/x', id(self.l3.x))])

    def test_namedparams_skip_uninit(self):
        namedparams = list(self.c2.namedparams(include_uninit=False))
        self.assertEqual([(name, id(p)) for name, p in namedparams],
                         [('/0/0/x', id(self.l1.x)),
                          ('/0/1/x', id(self.l2.x)),
                          ('/1/x', id(self.l3.x))])

    def test_links(self):
        links = list(self.c2.links())
        self.assertEqual([id(l) for l in links],
                         [id(l) for l in [self.c2,
                                          self.c1, self.l1, self.l2,
                                          self.l3]])

    def test_links_skipself(self):
        links = list(self.c2.links(skipself=True))
        self.assertEqual([id(l) for l in links],
                         [id(l) for l in [self.c1, self.l1, self.l2,
                                          self.l3]])

    def test_namedlinks(self):
        namedlinks = list(self.c2.namedlinks())
        self.assertEqual([(name, id(l)) for name, l in namedlinks],
                         [('/', id(self.c2)),
                          ('/0', id(self.c1)),
                          ('/0/0', id(self.l1)),
                          ('/0/1', id(self.l2)),
                          ('/1', id(self.l3))])

    def test_namedlinks_skipself(self):
        namedlinks = list(self.c2.namedlinks(skipself=True))
        self.assertEqual([(name, id(l)) for name, l in namedlinks],
                         [('/0', id(self.c1)),
                          ('/0/0', id(self.l1)),
                          ('/0/1', id(self.l2)),
                          ('/1', id(self.l3))])

    def test_children(self):
        self.assertEqual(tuple(id(c) for c in self.c2.children()),
                         (id(self.c1), id(self.l3)))

        self.assertEqual(tuple(id(c) for c in self.c1.children()),
                         (id(self.l1), id(self.l2)))

    def test_copyparams(self):
        l1 = chainer.Link()
        with l1.init_scope():
            l1.x = chainer.Parameter(shape=(2, 3))
            l1.y = chainer.Parameter()
        l2 = chainer.Link()
        with l2.init_scope():
            l2.x = chainer.Parameter(shape=2)
        l3 = chainer.Link()
        with l3.init_scope():
            l3.x = chainer.Parameter(shape=3)
        c1 = chainer.ChainList(l1, l2)
        c2 = chainer.ChainList(c1, l3)
        l1.x.data.fill(0)
        l2.x.data.fill(1)
        l3.x.data.fill(2)

        self.c2.copyparams(c2)

        numpy.testing.assert_array_equal(self.l1.x.data, l1.x.data)
        numpy.testing.assert_array_equal(self.l2.x.data, l2.x.data)
        numpy.testing.assert_array_equal(self.l3.x.data, l3.x.data)

    def test_zerograds(self):
        with testing.assert_warns(DeprecationWarning):
            self.c2.zerograds()
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.zeros(3))
        self.l1.y.initialize((2, 3))
        numpy.testing.assert_array_equal(self.l1.y.grad, numpy.zeros((2, 3)))

    def test_cleargrads(self):
        self.c2.cleargrads()
        self.assertIsNone(self.l1.x.grad)
        self.assertIsNone(self.l2.x.grad)
        self.assertIsNone(self.l3.x.grad)
        self.l1.y.initialize((2, 3))
        self.assertIsNone(self.l1.y.grad)

    def test_addgrads(self):
        l1 = chainer.Link()
        with l1.init_scope():
            l1.x = chainer.Parameter(shape=(2, 3))
            l1.y = chainer.Parameter(shape=(2, 3))
        l2 = chainer.Link()
        with l2.init_scope():
            l2.x = chainer.Parameter(shape=2)
        l3 = chainer.Link()
        with l3.init_scope():
            l3.x = chainer.Parameter(shape=3)
        c1 = chainer.ChainList(l1, l2)
        c2 = chainer.ChainList(c1, l3)
        l1.x.grad.fill(1)
        l2.x.grad.fill(2)
        l3.x.grad.fill(3)
        l1.y.grad.fill(4)

        self.l1.x.grad.fill(-1)
        self.l1.y.cleargrad()
        self.l2.x.grad.fill(-2)
        self.l3.x.grad.fill(-3)

        self.c2.addgrads(c2)
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l1.y.grad, l1.y.grad)
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.zeros(3))

    def test_serialize(self):
        l1 = chainer.Link()
        with l1.init_scope():
            l1.y = chainer.Parameter(shape=(1, 1))

        l2 = chainer.Link()
        with l2.init_scope():
            l2.x = chainer.Parameter(0, 2)
        c1 = chainer.ChainList(l1, l2)
        mocks = {'0': mock.MagicMock(), '1': mock.MagicMock()}
        serializer = mock.MagicMock()
        serializer.__getitem__.side_effect = lambda k: mocks[k]
        serializer.return_value = None
        mocks['0'].return_value = None
        mocks['1'].return_value = None
        c1.serialize(serializer)

        self.assertEqual(serializer.call_count, 0)
        self.assertEqual(serializer.__getitem__.call_count, 2)
        serializer.__getitem__.assert_any_call('0')
        serializer.__getitem__.assert_any_call('1')

        mocks['0'].assert_called_with('y', l1.y.data)
        mocks['1'].assert_called_with('x', l2.x.data)

    def test_count_params(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            assert self.c1.count_params() == 8
        assert len(w) == 1
        assert w[0].category is UserWarning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.c2.count_params()
        assert len(w) == 1
        assert w[0].category is UserWarning

        self.c2[0][0].y.initialize((2, 3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.c2.count_params()
        assert not w


class TestChainListRepeat(unittest.TestCase):

    def setUp(self):
        class ChainListForTest(chainer.ChainList):
            def __init__(self):
                super(ChainListForTest, self).__init__(chainer.Link())

            def forward(self):
                pass

        self.chainlist = ChainListForTest()
        self.link = self.chainlist[0]
        with self.link.init_scope():
            self.link.x = chainer.Parameter(
                chainer.initializers.Normal(), shape=(2, 3))

    def test_no_repeat(self):
        ret = self.chainlist.repeat(0)
        self.assertEqual(len(ret), 0)

    def test_repeat_with_share_mode(self):
        ret = self.chainlist.repeat(2, mode='share')
        self.assertEqual(len(ret), 2)
        self.assertIsNot(ret[0], self.chainlist)
        self.assertIsNot(ret[1], self.chainlist)
        self.assertIsNot(ret[0], ret[1])
        self.assertIsNot(ret[0][0], self.chainlist[0])
        self.assertIsNot(ret[1][0], self.chainlist[0])
        self.assertIsNot(ret[0][0], ret[1][0])
        self.assertIsNot(ret[0][0].x, self.chainlist[0].x)
        self.assertIsNot(ret[1][0].x, self.chainlist[0].x)
        self.assertIsNot(ret[0][0].x, ret[1][0].x)
        self.assertIs(ret[0][0].x.data, self.chainlist[0].x.data)
        self.assertIs(ret[0][0].x.data, ret[1][0].x.data)
        self.assertEqual(ret[0][0].x.shape, self.chainlist[0].x.shape)
        self.assertEqual(ret[0][0].x.shape, ret[1][0].x.shape)
        self.assertEqual(ret[0][0].x.dtype, self.chainlist[0].x.dtype)
        self.assertEqual(ret[0][0].x.dtype, ret[1][0].x.dtype)

    def test_repeat_with_copy_mode(self):
        ret = self.chainlist.repeat(2, mode='copy')
        self.assertEqual(len(ret), 2)
        self.assertIsNot(ret[0], self.chainlist)
        self.assertIsNot(ret[1], self.chainlist)
        self.assertIsNot(ret[0], ret[1])
        self.assertIsNot(ret[0][0], self.chainlist[0])
        self.assertIsNot(ret[1][0], self.chainlist[0])
        self.assertIsNot(ret[0][0], ret[1][0])
        self.assertIsNot(ret[0][0].x, self.chainlist[0].x)
        self.assertIsNot(ret[1][0].x, self.chainlist[0].x)
        self.assertIsNot(ret[0][0].x, ret[1][0].x)
        self.assertIsNot(ret[0][0].x.data, self.chainlist[0].x.data)
        self.assertIsNot(ret[1][0].x.data, self.chainlist[0].x.data)
        self.assertIsNot(ret[0][0].x.data, ret[1][0].x.data)
        self.assertTrue(numpy.array_equal(
            ret[0][0].x.data, self.chainlist[0].x.data))
        self.assertTrue(numpy.array_equal(
            ret[0][0].x.data, ret[1][0].x.data))
        self.assertEqual(ret[0][0].x.shape, self.chainlist[0].x.shape)
        self.assertEqual(ret[0][0].x.shape, ret[1][0].x.shape)
        self.assertEqual(ret[0][0].x.dtype, self.chainlist[0].x.dtype)
        self.assertEqual(ret[0][0].x.dtype, ret[1][0].x.dtype)

    def test_repeat_with_init_mode(self):
        ret = self.chainlist.repeat(2, mode='init')
        self.assertEqual(len(ret), 2)
        self.assertIsNot(ret[0], self.chainlist)
        self.assertIsNot(ret[1], self.chainlist)
        self.assertIsNot(ret[0], ret[1])
        self.assertIsNot(ret[0][0], self.chainlist[0])
        self.assertIsNot(ret[1][0], self.chainlist[0])
        self.assertIsNot(ret[0][0], ret[1][0])
        self.assertIsNot(ret[0][0].x, self.chainlist[0].x)
        self.assertIsNot(ret[1][0].x, self.chainlist[0].x)
        self.assertIsNot(ret[0][0].x, ret[1][0].x)
        self.assertIsNot(ret[0][0].x.data, self.chainlist[0].x.data)
        self.assertIsNot(ret[1][0].x.data, self.chainlist[0].x.data)
        self.assertIsNot(ret[0][0].x.data, ret[1][0].x.data)
        self.assertFalse(numpy.array_equal(
            ret[0][0].x.data, self.chainlist[0].x.data))
        self.assertFalse(numpy.array_equal(
            ret[1][0].x.data, self.chainlist[0].x.data))
        self.assertFalse(numpy.array_equal(
            ret[0][0].x.data, ret[1][0].x.data))
        self.assertEqual(ret[0][0].x.shape, self.chainlist[0].x.shape)
        self.assertEqual(ret[0][0].x.shape, ret[1][0].x.shape)
        self.assertEqual(ret[0][0].x.dtype, self.chainlist[0].x.dtype)
        self.assertEqual(ret[0][0].x.dtype, ret[1][0].x.dtype)


@attr.ideep
class TestIntel64(unittest.TestCase):

    def setUp(self):
        self.link = chainer.Link()
        shape = (2, 2)
        dtype = numpy.float32
        y_array = numpy.random.rand(*shape).astype(dtype)
        pa_array = numpy.random.rand(*shape).astype(dtype)
        ps_scalar = 2.4

        with self.link.init_scope():
            # Initialized parameter
            self.link.y = chainer.Parameter(y_array)
            # Uninitialized parameter
            self.link.v = chainer.Parameter()
            # Persistent ndarray
            self.link.add_persistent('pa', pa_array)
            # Persistent scalar
            self.link.add_persistent('ps', ps_scalar)
        self.y_array = y_array
        self.pa_array = pa_array
        self.ps_scalar = ps_scalar

    def test_cpu_to_intel64(self):
        link = self.link
        link.to_intel64()
        assert isinstance(link.device, backend.Intel64Device)

        # Arrays should be converted to ideep.mdarray

        # Initialized parameter
        assert isinstance(link.y.data, intel64.ideep.mdarray)
        _assert_variable_array_equal(link.y, self.y_array)
        # Uninitialized parameter
        assert link.v.data is None
        # Persistent ndarray
        assert isinstance(link.pa, intel64.ideep.mdarray)
        _assert_arrays_equal(link.pa, self.pa_array)
        # Persistent scalar
        assert link.ps == self.ps_scalar

    def test_intel64_to_intel64(self):
        link = self.link
        link.to_intel64()
        prev_y = link.y
        prev_v = link.v
        prev_pa = link.pa
        prev_ps = link.ps
        link.to_intel64()
        assert isinstance(link.device, backend.Intel64Device)

        # Everything should be left untouched

        # Initialized parameter
        assert link.y is prev_y
        # Uninitialized parameter
        assert link.v is prev_v
        # Persistent ndarray
        assert link.pa is prev_pa
        # Persistent scalar
        assert link.ps is prev_ps

    @attr.gpu
    def test_gpu_to_intel64(self):
        link = self.link
        link.to_gpu()
        assert link.device.device == cuda.Device(0)
        link.to_intel64()
        assert isinstance(link.device, backend.Intel64Device)

        # Arrays should be converted to ideep.mdarray

        # Initialized parameter
        assert isinstance(link.y.data, intel64.ideep.mdarray)
        _assert_variable_array_equal(link.y, self.y_array)
        # Uninitialized parameter
        assert link.v.data is None
        # Persistent ndarray
        assert isinstance(link.pa, intel64.ideep.mdarray)
        _assert_arrays_equal(link.pa, self.pa_array)
        # Persistent scalar
        assert link.ps == self.ps_scalar

    @attr.gpu
    def test_intel64_to_gpu(self):
        link = self.link
        link.to_intel64()
        assert isinstance(link.device, backend.Intel64Device)
        link.to_gpu()
        assert link.device.device == cuda.Device(0)

        # Arrays should be converted to cupy.ndarray

        # Initialized parameter
        assert isinstance(link.y.data, cuda.cupy.ndarray)
        _assert_variable_array_equal(link.y, self.y_array)
        # Uninitialized parameter
        assert link.v.data is None
        # Persistent ndarray
        assert isinstance(link.pa, cuda.ndarray)
        _assert_arrays_equal(link.pa, self.pa_array)
        # Persistent scalar
        assert link.ps == self.ps_scalar

    def test_intel64_to_cpu(self):
        link = self.link
        link.to_intel64()
        assert isinstance(link.device, backend.Intel64Device)
        link.to_cpu()
        assert isinstance(link.device, backend.CpuDevice)

        # Arrays should be converted to numpy.ndarray

        # Initialized parameter
        assert isinstance(link.y.data, numpy.ndarray)
        _assert_variable_array_equal(link.y, self.y_array)
        # Uninitialized parameter
        assert link.v.data is None
        # Persistent ndarray
        assert isinstance(link.pa, numpy.ndarray)
        _assert_arrays_equal(link.pa, self.pa_array)
        # Persistent scalar
        assert link.ps == self.ps_scalar

    def test_cpu_to_intel64_unsupported(self):
        # Test for persistents that cannot be transferred to iDeep.
        with self.link.init_scope():
            self.link.no_ideep = numpy.ones((2, 2, 2), numpy.float32)
            self.link.register_persistent('no_ideep')
        self.link.to_intel64()
        assert isinstance(self.link.no_ideep, numpy.ndarray)

    @attr.gpu
    def test_gpu_to_intel64_unsupported(self):
        # Test for persistents that cannot be transferred to iDeep.
        with self.link.init_scope():
            self.link.no_ideep = cuda.cupy.ones((2, 2, 2), numpy.float32)
            self.link.register_persistent('no_ideep')
        self.link.to_intel64()
        assert isinstance(self.link.no_ideep, numpy.ndarray)


@attr.chainerx
class TestToChainerX(unittest.TestCase):

    def setUp(self):
        self.link = chainer.Link()
        shape = (2, 2)
        dtype = numpy.float32
        y_array = numpy.random.rand(*shape).astype(dtype)
        pa_array = numpy.random.rand(*shape).astype(dtype)
        ps_scalar = 2.4

        with self.link.init_scope():
            # Initialized parameter
            self.link.y = chainer.Parameter(y_array)
            # Uninitialized parameter
            self.link.v = chainer.Parameter()
            # Persistent ndarray
            self.link.add_persistent('pa', pa_array)
            # Persistent scalar
            self.link.add_persistent('ps', ps_scalar)
        self.y_array = y_array
        self.pa_array = pa_array
        self.ps_scalar = ps_scalar

    def test_chainerx_to_chainerx(self):
        link = self.link
        link.to_chainerx()
        prev_y = link.y
        prev_v = link.v
        prev_pa = link.pa
        prev_ps = link.ps
        link.to_chainerx()
        assert link.device.device == chainerx.get_device('native:0')

        # Everything should be left untouched

        # Initialized parameter
        assert link.y is prev_y
        # Uninitialized parameter
        assert link.v is prev_v
        # Persistent ndarray
        assert link.pa is prev_pa
        # Persistent scalar
        assert link.ps is prev_ps

    def test_cpu_to_chainerx(self):
        link = self.link
        link.to_chainerx()

        # Initialized parameter
        assert isinstance(link.y.data, chainerx.ndarray)
        assert link.y.data.device.backend.name == 'native'
        assert link.y.data.device.index == 0
        _assert_variable_array_equal(link.y, self.y_array)
        # Uninitialized parameter
        assert link.v.data is None
        # Persistent ndarray
        assert isinstance(link.pa, chainerx.ndarray)
        _assert_arrays_equal(link.pa, self.pa_array)
        # Persistent scalar
        assert link.ps == self.ps_scalar

    @attr.gpu
    def test_gpu_to_chainerx(self):
        link = self.link
        link.to_gpu()
        assert link.device.device == cuda.Device(0)
        link.to_chainerx()
        assert link.device.device == chainerx.get_device('cuda:0')

        # Arrays should be converted to chainerx.ndarray

        # Initialized parameter
        assert isinstance(link.y.data, chainerx.ndarray)
        assert link.y.data.device.backend.name == 'cuda'
        assert link.y.data.device.index == 0
        _assert_variable_array_equal(link.y, self.y_array)
        # Uninitialized parameter
        assert link.v.data is None
        # Persistent ndarray
        assert isinstance(link.pa, chainerx.ndarray)
        _assert_arrays_equal(link.pa, self.pa_array)
        # Persistent scalar
        assert link.ps == self.ps_scalar

    # TODO(niboshi): Add other test variations


class TestToDevice(unittest.TestCase):
    def setUp(self):
        self.link = chainer.Link()
        shape = (2, 2)
        dtype = numpy.float32
        y_array = numpy.random.rand(*shape).astype(dtype)
        pa_array = numpy.random.rand(*shape).astype(dtype)
        ps_scalar = 2.4

        with self.link.init_scope():
            # Initialized parameter
            self.link.y = chainer.Parameter(y_array)
            # Uninitialized parameter
            self.link.v = chainer.Parameter()
            # Persistent ndarray
            self.link.add_persistent('pa', pa_array)
            # Persistent scalar
            self.link.add_persistent('ps', ps_scalar)
        self.y_array = y_array
        self.pa_array = pa_array
        self.ps_scalar = ps_scalar

        if cuda.available:
            self.current_device_id = cuda.cupy.cuda.get_device_id()

    def check_to_device(self, device_spec, expected_ndarray_type):
        link = self.link

        link.to_device(device_spec)

        # Initialized parameter
        assert isinstance(link.y.data, expected_ndarray_type)
        _assert_variable_array_equal(link.y, self.y_array)
        # Uninitialized parameter
        assert link.v.data is None
        # Persistent ndarray
        assert isinstance(link.pa, expected_ndarray_type)
        _assert_arrays_equal(link.pa, self.pa_array)
        # Persistent scalar
        assert link.ps == self.ps_scalar

        return link

    def test_to_device_numpy(self):
        link = self.check_to_device(numpy, numpy.ndarray)
        assert isinstance(link.device, backend.CpuDevice)

    @attr.gpu
    def test_to_device_cupy(self):
        link = self.check_to_device((cuda.cupy, 0), cuda.ndarray)
        assert link.device.device == cuda.Device(0)

    @attr.chainerx
    def test_to_device_chainerx(self):
        link = self.check_to_device('native:0', chainerx.ndarray)
        assert link.device.device == chainerx.get_device('native:0')


class TestCallMethod(unittest.TestCase):

    def setUp(self):
        class Model(chainer.Chain):
            def __init__(self):
                super(Model, self).__init__()

        self.model = Model()

    def test_has_forward_no_call(self):
        self.model.forward = mock.MagicMock()
        self.model(0)  # model.forward is called
        self.model.forward.assert_called_once()

    def test_has_call_and_forward(self):
        self.model.__call__ = mock.MagicMock()
        self.model.forward = mock.MagicMock()
        self.model(0)  # Link.__call__ is called
        self.model.forward.assert_called_with(0)
        self.model.__call__.assert_not_called()

    def test_has_call_no_forward(self):
        class Model(chainer.Chain):
            def __init__(self):
                super(Model, self).__init__()
                self.mock = mock.MagicMock()

            def __call__(self, x):
                self.mock(x)

        model = Model()
        model(0)  # model.__call__ is called
        model.mock.assert_called_with(0)

    def test_no_call_no_forward(self):
        with self.assertRaises(AttributeError):
            self.model(0)


testing.run_module(__name__, __file__)
