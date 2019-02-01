import os
import platform
import tempfile
import unittest

import mock
import numpy
import six

from chainer import cuda
from chainer import functions
from chainer import links
from chainer import sequential
from chainer import testing
from chainer.testing import attr
from chainer import variable


class TestSequential(unittest.TestCase):

    def setUp(self):
        self.l1 = links.Linear(None, 3)
        self.l2 = links.Linear(3, 2)
        self.l3 = links.Linear(2, 3)
        # s1: l1 -> l2
        self.s1 = sequential.Sequential(self.l1)
        self.s1.append(self.l2)
        # s2: s1 (l1 -> l2) -> l3
        self.s2 = sequential.Sequential(self.s1)
        self.s2.append(self.l3)

    def test_init(self):
        self.assertIs(self.s1[0], self.l1)
        self.assertEqual(self.l1.name, '0')
        self.assertIs(self.s2[0], self.s1)
        self.assertEqual(self.s1.name, '0')
        with self.assertRaises(ValueError):
            sequential.Sequential(0)

    def test_append(self):
        self.assertIs(self.s2[1], self.l3)
        self.assertEqual(self.l2.name, '1')

    def test_iter(self):
        links = list(self.s2)
        self.assertEqual(2, len(links))
        self.assertIs(links[0], self.s1)
        self.assertIs(links[1], self.l3)

    def test_len(self):
        self.assertIs(len(self.s1), 2)
        self.assertIs(len(self.s2), 2)

    def test_copy(self):
        s2 = self.s2.copy()

        self.assertIs(s2.name, None)
        self.assertIsInstance(s2._children, list)
        self.assertIsNot(s2[0], self.s1)
        self.assertEqual(s2[0].name, '0')
        self.assertIsInstance(s2[0]._children, list)
        self.assertIsNot(s2[0][0], self.l1)
        self.assertEqual(s2[0][0].name, '0')
        self.assertIsNot(s2[0][0].b, self.l1.b)
        self.assertIs(s2[0][0].b.data, self.l1.b.data)
        self.assertIs(s2[0][0].b.grad, None)

        self.assertIsNot(s2[0][1], self.l2)
        self.assertEqual(s2[0][1].name, '1')
        self.assertIsNot(s2[0][1].W, self.l2.W)
        self.assertIs(s2[0][1].W.data, self.l2.W.data)
        self.assertIs(s2[0][1].W.grad, None)

        self.assertIsNot(s2[1], self.l3)
        self.assertEqual(s2[1].name, '1')
        self.assertIsNot(s2[1].W, self.l3.W)
        self.assertIs(s2[1].W.data, self.l3.W.data)
        self.assertIs(s2[1].W.grad, None)

    def test_copy_with_nonparametric_function(self):
        self.s1.insert(1, functions.relu)
        # l1 -> relu -> l2

        # The default copy mode is 'share'
        s1 = self.s1.copy()
        self.assertIsNot(s1[0], self.s1[0])  # l1
        self.assertIs(s1[1], self.s1[1])  # relu
        self.assertIsNot(s1[2], self.s1[2])  # l2

        # parameters of l1
        self.assertIsNot(s1[0].W, self.s1[0].W)
        self.assertIsNot(s1[0].b, self.s1[0].b)
        # W of the first link has not been initialized
        self.assertIs(s1[0].W.array, None)
        self.assertIs(s1[0].W.grad, None)
        # The bias is initialized
        self.assertIs(s1[0].b.array, self.s1[0].b.array)
        self.assertIs(s1[0].b.grad, None)

        # The copied Function should be identical
        self.assertIs(s1[1], self.s1[1])

        # parameters of l2
        self.assertIsNot(s1[2].W, self.s1[2].W)
        self.assertIsNot(s1[2].b, self.s1[2].b)
        self.assertIs(s1[2].W.array, self.s1[2].W.array)
        self.assertIs(s1[2].W.grad, None)
        self.assertIs(s1[2].b.array, self.s1[2].b.array)
        self.assertIs(s1[2].b.grad, None)

    @attr.gpu
    def test_copy_and_send_to_gpu(self):
        s2 = self.s2.copy()
        self.s2.to_gpu()
        self.assertIsInstance(self.s2[0][0].b.data, cuda.cupy.ndarray)
        self.assertIsInstance(self.s2[0][1].W.data, cuda.cupy.ndarray)
        self.assertIsInstance(s2[0][0].b.data, numpy.ndarray)
        self.assertIsInstance(s2[0][1].W.data, numpy.ndarray)

    @attr.gpu
    def test_copy_and_send_to_gpu_2(self):
        s2 = self.s2.copy()
        s2.to_gpu()
        self.assertIsInstance(self.s2[0][0].b.data, numpy.ndarray)
        self.assertIsInstance(self.s2[0][1].W.data, numpy.ndarray)
        self.assertIsInstance(s2[0][0].b.data, cuda.cupy.ndarray)
        self.assertIsInstance(s2[0][1].W.data, cuda.cupy.ndarray)

    @attr.multi_gpu(2)
    def test_copy_and_send_to_gpu_multi(self):
        s2 = self.s2.copy()
        self.s2.to_gpu(0)
        s2.to_gpu(1)
        self.assertEqual(self.s2[0][0].b.data.device.id, 0)
        self.assertEqual(self.s2[0][1].W.data.device.id, 0)
        self.assertEqual(s2[0][0].b.data.device.id, 1)
        self.assertEqual(s2[0][1].W.data.device.id, 1)

    def test_to_cpu_on_cpu(self):
        x1 = self.l1.b.data
        gx1 = self.l1.b.grad
        x2 = self.l2.W.data
        gx2 = self.l2.W.grad
        x3 = self.l3.W.data
        gx3 = self.l3.W.grad

        self.s2.to_cpu()

        self.assertIs(self.l1.b.data, x1)
        self.assertIs(self.l1.b.grad, gx1)
        self.assertIs(self.l2.W.data, x2)
        self.assertIs(self.l2.W.grad, gx2)
        self.assertIs(self.l3.W.data, x3)
        self.assertIs(self.l3.W.grad, gx3)

    @attr.gpu
    def test_to_cpu(self):
        self.s2.to_gpu()
        self.s2.to_cpu()
        self.assertIs(self.s2.xp, numpy)
        self.assertIs(self.s1.xp, numpy)
        self.assertIs(self.l1.xp, numpy)
        self.assertIs(self.l2.xp, numpy)
        self.assertIs(self.l3.xp, numpy)
        self.assertIsInstance(self.l1.b.data, numpy.ndarray)
        self.assertIsInstance(self.l1.b.grad, numpy.ndarray)
        self.assertIsInstance(self.l2.W.data, numpy.ndarray)
        self.assertIsInstance(self.l2.W.grad, numpy.ndarray)
        self.assertIsInstance(self.l3.W.data, numpy.ndarray)
        self.assertIsInstance(self.l3.W.grad, numpy.ndarray)

    @attr.gpu
    def test_to_gpu(self):
        cupy = cuda.cupy
        self.s2.to_gpu()
        self.assertIs(self.s2.xp, cupy)
        self.assertIs(self.s1.xp, cupy)
        self.assertIs(self.l1.xp, cupy)
        self.assertIs(self.l2.xp, cupy)
        self.assertIs(self.l3.xp, cupy)
        self.assertIsInstance(self.l1.b.data, cupy.ndarray)
        self.assertIsInstance(self.l1.b.grad, cupy.ndarray)
        self.assertIsInstance(self.l2.W.data, cupy.ndarray)
        self.assertIsInstance(self.l2.W.grad, cupy.ndarray)
        self.assertIsInstance(self.l3.W.data, cupy.ndarray)
        self.assertIsInstance(self.l3.W.grad, cupy.ndarray)

    def test_params(self):
        params = list(self.s2.params())
        self.assertEqual({id(p) for p in params},
                         {id(self.l1.W), id(self.l1.b),
                          id(self.l2.W), id(self.l2.b),
                          id(self.l3.W), id(self.l3.b)})

    def test_params_skip_uninit(self):
        params = list(self.s2.params(include_uninit=False))
        self.assertEqual({id(p) for p in params},
                         {id(self.l1.b), id(self.l2.W), id(self.l2.b),
                          id(self.l3.W), id(self.l3.b)})

    def test_namedparams(self):
        namedparams = list(self.s2.namedparams())
        self.assertEqual({(name, id(p)) for name, p in namedparams},
                         {('/0/0/W', id(self.l1.W)),
                          ('/0/0/b', id(self.l1.b)),
                          ('/0/1/W', id(self.l2.W)),
                          ('/0/1/b', id(self.l2.b)),
                          ('/1/W', id(self.l3.W)),
                          ('/1/b', id(self.l3.b))})

    def test_namedparams_skip_uninit(self):
        namedparams = list(self.s2.namedparams(include_uninit=False))
        self.assertEqual({(name, id(p)) for name, p in namedparams},
                         {('/0/0/b', id(self.l1.b)),
                          ('/0/1/W', id(self.l2.W)),
                          ('/0/1/b', id(self.l2.b)),
                          ('/1/W', id(self.l3.W)),
                          ('/1/b', id(self.l3.b))})

    def test_links(self):
        links = list(self.s2.links())
        self.assertEqual({id(l) for l in links},
                         {id(l) for l in [self.l1, self.l2, self.l3,
                                          self.s1, self.s2]})

    def test_links_skipself(self):
        links = list(self.s2.links(skipself=True))
        self.assertEqual({id(l) for l in links},
                         {id(l) for l in [self.l1, self.l2, self.l3, self.s1]})

    def test_namedlinks(self):
        namedlinks = list(self.s2.namedlinks())
        self.assertEqual({(name, id(l)) for name, l in namedlinks},
                         {('/', id(self.s2)),
                          ('/0', id(self.s1)),
                          ('/0/0', id(self.l1)),
                          ('/0/1', id(self.l2)),
                          ('/1', id(self.l3))})

    def test_namedlinks_skipself(self):
        namedlinks = list(self.s2.namedlinks(skipself=True))
        self.assertEqual({(name, id(l)) for name, l in namedlinks},
                         {('/0', id(self.s1)),
                          ('/0/0', id(self.l1)),
                          ('/0/1', id(self.l2)),
                          ('/1', id(self.l3))})

    def test_children(self):
        self.assertEqual(tuple(id(c) for c in self.s2.children()),
                         (id(self.s1), id(self.l3)))

        self.assertEqual(tuple(id(c) for c in self.s1.children()),
                         (id(self.l1), id(self.l2)))

    def test_copyparams(self):
        l1 = links.Linear(None, 3)
        l2 = links.Linear(3, 2)
        l3 = links.Linear(2, 3)
        s1 = sequential.Sequential(l1, l2)
        s2 = sequential.Sequential(s1, l3)
        l1.b.data.fill(0)
        l2.W.data.fill(1)
        l2.b.data.fill(2)
        l3.W.data.fill(3)
        l3.b.data.fill(4)

        self.s2.copyparams(s2)

        numpy.testing.assert_array_equal(self.l1.b.data, l1.b.data)
        numpy.testing.assert_array_equal(self.l2.W.data, l2.W.data)
        numpy.testing.assert_array_equal(self.l2.b.data, l2.b.data)
        numpy.testing.assert_array_equal(self.l3.W.data, l3.W.data)
        numpy.testing.assert_array_equal(self.l3.b.data, l3.b.data)

    def test_zerograds(self):
        with testing.assert_warns(DeprecationWarning):
            self.s2.zerograds()
            numpy.testing.assert_array_equal(self.l1.b.grad, numpy.zeros((3,)))
            numpy.testing.assert_array_equal(
                self.l2.W.grad, numpy.zeros((2, 3)))
            numpy.testing.assert_array_equal(
                self.l3.W.grad, numpy.zeros((3, 2)))
            self.l1.W.initialize((3, 2))
            numpy.testing.assert_array_equal(
                self.l1.W.grad, numpy.zeros((3, 2)))

    def test_cleargrads(self):
        self.s2.cleargrads()
        self.assertIsNone(self.l1.b.grad)
        self.assertIsNone(self.l2.W.grad)
        self.assertIsNone(self.l2.b.grad)
        self.assertIsNone(self.l3.W.grad)
        self.assertIsNone(self.l3.b.grad)
        self.l1.W.initialize((2, 3))
        self.assertIsNone(self.l1.W.grad)

    def test_addgrads(self):
        l1 = links.Linear(2, 3)
        l2 = links.Linear(3, 2)
        l3 = links.Linear(2, 3)
        s1 = sequential.Sequential(l1, l2)
        s2 = sequential.Sequential(s1, l3)
        l1.b.grad.fill(1)
        l2.W.grad.fill(2)
        l2.b.grad.fill(3)
        l3.W.grad.fill(4)
        l3.b.grad.fill(5)
        l1.W.grad.fill(6)

        self.l1.b.grad.fill(-1)
        self.l2.W.grad.fill(-2)
        self.l2.b.grad.fill(-3)
        self.l3.W.grad.fill(-4)
        self.l3.b.grad.fill(-5)
        self.l1.W.cleargrad()

        self.s2.addgrads(s2)
        numpy.testing.assert_array_equal(self.l1.b.grad, numpy.zeros((3,)))
        numpy.testing.assert_array_equal(self.l1.W.grad, l1.W.grad)
        numpy.testing.assert_array_equal(self.l2.W.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l2.b.grad, numpy.zeros((2,)))
        numpy.testing.assert_array_equal(self.l3.W.grad, numpy.zeros((3, 2)))
        numpy.testing.assert_array_equal(self.l3.b.grad, numpy.zeros((3,)))

    def test_serialize(self):
        l1 = links.Linear(None, 1)
        l2 = links.Linear(None, 3)
        with l2.init_scope():
            l2.x = variable.Parameter(0, 2)
        s1 = sequential.Sequential(l1, l2)
        mocks = {'0': mock.MagicMock(), '1': mock.MagicMock()}
        serializer = mock.MagicMock()
        serializer.__getitem__.side_effect = lambda k: mocks[k]
        serializer.return_value = None
        mocks['0'].return_value = None
        mocks['1'].return_value = None
        s1.serialize(serializer)

        self.assertEqual(serializer.call_count, 0)
        self.assertEqual(serializer.__getitem__.call_count, 2)
        serializer.__getitem__.assert_any_call('0')
        serializer.__getitem__.assert_any_call('1')

        mocks['0'].assert_any_call('W', None)
        mocks['0'].assert_any_call('b', l1.b.data)
        mocks['1'].assert_any_call('W', None)
        mocks['1'].assert_any_call('b', l2.b.data)
        mocks['1'].assert_any_call('x', l2.x.data)

    def test_getitem(self):
        self.assertIs(self.s1[0], self.l1)

    def test_delitem(self):
        del self.s1[0]
        self.assertIsNot(self.s1[0], self.l1)
        self.assertIs(self.s1[0], self.l2)

    def test_reversed(self):
        layers = list(reversed(self.s2))
        self.assertIs(layers[0], self.l3)
        self.assertIs(layers[1], self.s1)

    def test_contains(self):
        self.assertTrue(self.l1 in self.s1)
        self.assertTrue(self.l2 in self.s1)
        self.assertTrue(self.s1 in self.s2)
        self.assertTrue(self.l3 in self.s2)
        self.assertFalse(self.l3 in self.s1)
        self.assertFalse(self.l2 in self.s2)

    def test_add(self):
        l1 = links.Linear(3, 2)
        l2 = functions.relu
        other = sequential.Sequential(l1, l2)
        added = self.s2 + other
        self.assertEqual(len(added), 4)
        self.assertIs(added[0], self.s1)
        self.assertIs(added[1], self.l3)
        self.assertIs(added[2], l1)
        self.assertIs(added[3], l2)
        with self.assertRaises(ValueError):
            self.s2 + 0

    def test_iadd(self):
        l4 = links.Linear(3, 1)
        self.s2 += sequential.Sequential(l4)
        self.assertIs(self.s2[0], self.s1)
        self.assertIs(self.s2[1], self.l3)
        self.assertIs(self.s2[2], l4)
        with self.assertRaises(ValueError):
            self.s2 += 0

    def test_call(self):
        l1 = mock.MagicMock()
        l2 = mock.MagicMock()
        l3 = mock.MagicMock()
        model = sequential.Sequential(l1, l2, l3)
        x = numpy.arange(2).reshape(1, 2).astype('f')
        y = model(x)
        l1.assert_called_once()
        l2.assert_called_once()
        l3.assert_called_once()
        y = self.s1(x)
        self.assertIs(y.creator.inputs[1].data, self.l2.W.data)

    def test_call_with_multiple_inputs(self):
        model = sequential.Sequential(
            lambda x, y: (x * 2, y * 3, x + y),
            lambda x, y, z: x + y + z
        )
        y = model(2, 3)
        self.assertEqual(y, 18)

    def test_extend(self):
        l1 = links.Linear(3, 2)
        l2 = links.Linear(2, 3)
        s3 = sequential.Sequential(l1, l2)
        self.s2.extend(s3)
        self.assertEqual(len(self.s2), 4)
        self.assertIs(self.s2[2], s3[0])
        self.assertIs(self.s2[3], s3[1])

    def test_insert(self):
        l1 = links.Linear(3, 3)
        self.s1.insert(1, l1)
        self.assertEqual(len(self.s1), 3)
        self.assertIs(self.s1[1], l1)

    def test_remove(self):
        self.s2.remove(self.s1)
        self.assertEqual(len(self.s2), 1)
        self.assertIs(self.s2[0], self.l3)

    def test_remove_by_layer_type(self):
        self.s2.insert(2, functions.relu)
        self.s2.remove_by_layer_type('Linear')
        self.assertEqual(len(self.s2), 2)
        self.assertIs(self.s2[0], self.s1)
        self.assertIs(self.s2[1], functions.relu)

    def test_pop(self):
        l3 = self.s2.pop(1)
        self.assertIs(l3, self.l3)
        self.assertEqual(len(self.s2), 1)

    def test_clear(self):
        self.s2.clear()
        self.assertEqual(len(self.s2), 0)

    def test_index(self):
        self.assertEqual(self.s2.index(self.s1), 0)
        self.assertEqual(self.s2.index(self.l3), 1)

    def test_count(self):
        self.s2.insert(1, functions.relu)
        self.s2.insert(3, functions.relu)
        self.assertEqual(self.s2.count(functions.relu), 2)
        self.assertEqual(self.s2.count(self.s1), 1)
        self.assertEqual(self.s2.count(self.l3), 1)
        self.s2.append(self.l3)
        self.assertEqual(self.s2.count(self.l3), 2)

    def test_count_by_layer_type(self):
        self.assertEqual(self.s2.count_by_layer_type('Linear'), 1)
        self.s2.insert(1, functions.relu)
        self.s2.insert(3, functions.relu)
        self.assertEqual(self.s2.count_by_layer_type('relu'), 2)

    def test_pickle_without_lambda(self):
        fd, path = tempfile.mkstemp()
        six.moves.cPickle.dump(self.s2, open(path, 'wb'))
        s2 = six.moves.cPickle.load(open(path, 'rb'))
        self.assertEqual(len(s2), len(self.s2))
        numpy.testing.assert_array_equal(s2[0][0].b.data, self.s2[0][0].b.data)
        numpy.testing.assert_array_equal(s2[0][1].W.data, self.s2[0][1].W.data)
        numpy.testing.assert_array_equal(s2[0][1].b.data, self.s2[0][1].b.data)
        numpy.testing.assert_array_equal(s2[1].W.data, self.s2[1].W.data)
        numpy.testing.assert_array_equal(s2[1].b.data, self.s2[1].b.data)
        for l1, l2 in zip(s2, self.s2):
            self.assertIsNot(l1, l2)
        os.close(fd)
        os.remove(path)

    def test_pickle_with_lambda(self):
        self.s2.append(lambda x: x)
        with self.assertRaises(ValueError):
            with tempfile.TemporaryFile() as fp:
                six.moves.cPickle.dump(self.s2, fp)

    def test_repr(self):
        bits, pl = platform.architecture()
        self.assertEqual(
            str(self.s1),
            '0\tLinear\tW(None)\tb{}\t\n'
            '1\tLinear\tW{}\tb{}\t\n'.format(
                self.s1[0].b.shape, self.s1[1].W.shape, self.s1[1].b.shape))

    def test_repeat_with_init(self):
        # s2 ((l1 -> l2) -> l3) -> s2 ((l1 -> l2) -> l3)
        ret = self.s2.repeat(2)
        self.assertIsNot(ret[0], self.s2)
        self.assertIs(type(ret[0]), type(self.s2))
        self.assertIsNot(ret[1], self.s2)
        self.assertIs(type(ret[1]), type(self.s2))

        # b is filled with 0, so they should have the same values
        numpy.testing.assert_array_equal(
            ret[0][0][0].b.array, ret[1][0][0].b.array)
        # W is initialized randomly, so they should be different
        self.assertFalse(
            numpy.array_equal(ret[0][1].W.array, self.l3.W.array))
        # And the object should also be different
        self.assertIsNot(ret[0][1].W.array, self.l3.W.array)
        # Repeated elements should be different objects
        self.assertIsNot(ret[0], ret[1])
        # Also for the arrays
        self.assertIsNot(ret[0][1].W.array, ret[1][1].W.array)
        # And values should be different
        self.assertFalse(
            numpy.array_equal(ret[0][1].W.array, ret[1][1].W.array))

        self.assertEqual(len(ret), 2)
        ret = self.s2.repeat(0, mode='init')
        self.assertEqual(len(ret), 0)

    def test_repeat_with_copy(self):
        # s2 ((l1 -> l2) -> l3) -> s2 ((l1 -> l2) -> l3)
        ret = self.s2.repeat(2, mode='copy')
        self.assertIsNot(ret[0], self.s2)
        self.assertIs(type(ret[0]), type(self.s2))
        self.assertIsNot(ret[1], self.s2)
        self.assertIs(type(ret[1]), type(self.s2))
        self.assertIsNot(ret[0], ret[1])

        # b is filled with 0, so they should have the same values
        numpy.testing.assert_array_equal(
            ret[0][0][0].b.array, ret[1][0][0].b.array)
        # W is shallowy copied, so the values should be same
        numpy.testing.assert_array_equal(ret[0][1].W.array, self.l3.W.array)
        # But the object should be different
        self.assertIsNot(ret[0][1].W.array, self.l3.W.array)
        # Repeated elements should be different objects
        self.assertIsNot(ret[0][0], ret[1][0])
        # Also for the arrays
        self.assertIsNot(ret[0][1].W.array, ret[1][1].W.array)
        # But the values should be same
        numpy.testing.assert_array_equal(ret[0][1].W.array, ret[1][1].W.array)

        self.assertEqual(len(ret), 2)
        ret = self.s2.repeat(0, mode='copy')
        self.assertEqual(len(ret), 0)

    def test_repeat_with_share(self):
        # s2 ((l1 -> l2) -> l3) -> s2 ((l1 -> l2) -> l3)
        ret = self.s2.repeat(2, mode='share')
        self.assertIsNot(ret[0], self.s2)
        self.assertIs(type(ret[0]), type(self.s2))
        self.assertIsNot(ret[1], self.s2)
        self.assertIs(type(ret[1]), type(self.s2))

        # b is filled with 0, so they should have the same values
        numpy.testing.assert_array_equal(
            ret[0][0][0].b.data, ret[1][0][0].b.data)
        # W is shallowy copied, so the values should be same
        numpy.testing.assert_array_equal(ret[0][1].W.array, self.l3.W.array)
        numpy.testing.assert_array_equal(ret[1][1].W.array, self.l3.W.array)
        # And the object should also be same
        self.assertIs(ret[0][1].W.array, self.l3.W.array)
        self.assertIs(ret[1][1].W.array, self.l3.W.array)
        # Repeated element itself should be different
        self.assertIsNot(ret[0], ret[1])

        self.assertEqual(len(ret), 2)
        ret = self.s2.repeat(0, mode='share')
        self.assertEqual(len(ret), 0)

    def test_flatten(self):
        flattened_s2 = self.s2.flatten()
        self.assertIs(flattened_s2[0], self.l1)
        self.assertIs(flattened_s2[1], self.l2)
        self.assertIs(flattened_s2[2], self.l3)


testing.run_module(__name__, __file__)
