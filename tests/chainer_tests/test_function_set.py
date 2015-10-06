import unittest

import numpy as np
import six
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import testing
from chainer.testing import attr


class MockFunction(chainer.Link, chainer.Function):

    def __init__(self, shape):
        super(MockFunction, self).__init__()
        p = chainer.Variable(np.zeros(shape, dtype='f'))
        p.grad = np.ones(shape, dtype='f')
        self.params['p'] = p


class TestNestedFunctionSet(unittest.TestCase):

    def setUp(self):
        self.fs1 = chainer.FunctionSet(
            a=MockFunction((1, 2)))
        self.fs2 = chainer.FunctionSet(
            fs1=self.fs1,
            b=MockFunction((3, 4)))

    def test_collect_parameters(self):
        p_b = np.zeros((3, 4)).astype(np.float32)
        p_a = np.zeros((1, 2)).astype(np.float32)
        gp_b = np.ones((3, 4)).astype(np.float32)
        gp_a = np.ones((1, 2)).astype(np.float32)

        actual = (self.fs2.parameters, self.fs2.gradients)
        self.assertTrue(list(map(len, actual)) == [2, 2])
        self.assertTrue((actual[0][0] == p_b).all())
        self.assertTrue((actual[0][1] == p_a).all())
        self.assertTrue((actual[1][0] == gp_b).all())
        self.assertTrue((actual[1][1] == gp_a).all())

    def test_pickle_cpu(self):
        fs2_serialized = pickle.dumps(self.fs2)
        fs2_loaded = pickle.loads(fs2_serialized)
        self.assertTrue(
            (self.fs2.b.params['p'].data ==
             fs2_loaded.b.params['p'].data).all())
        self.assertTrue(
            (self.fs2.fs1.a.params['p'].data ==
             fs2_loaded.fs1.a.params['p'].data).all())

    @attr.gpu
    def test_pickle_gpu(self):
        self.fs2.to_gpu()
        fs2_serialized = pickle.dumps(self.fs2)
        fs2_loaded = pickle.loads(fs2_serialized)
        fs2_loaded.to_cpu()
        self.fs2.to_cpu()

        self.assertTrue(
            (self.fs2.b.params['p'].data ==
             fs2_loaded.b.params['p'].data).all())
        self.assertTrue(
            (self.fs2.fs1.a.params['p'].data ==
             fs2_loaded.fs1.a.params['p'].data).all())


class TestFunctionSet(unittest.TestCase):

    def setUp(self):
        self.fs = chainer.FunctionSet(
            a=F.Linear(3, 2),
            b=F.Linear(3, 2)
        )

    def check_equal_fs(self, fs1, fs2):
        self.assertTrue(
            (fs1.a.params['W'].data == fs2.a.params['W'].data).all())
        self.assertTrue(
            (fs1.a.params['b'].data == fs2.a.params['b'].data).all())
        self.assertTrue(
            (fs1.b.params['W'].data == fs2.b.params['W'].data).all())
        self.assertTrue(
            (fs1.b.params['b'].data == fs2.b.params['b'].data).all())

    def test_pickle_cpu(self):
        s = pickle.dumps(self.fs)
        fs2 = pickle.loads(s)
        self.check_equal_fs(self.fs, fs2)

    @attr.gpu
    def test_pickle_gpu(self):
        self.fs.to_gpu()
        s = pickle.dumps(self.fs)
        fs2 = pickle.loads(s)

        self.fs.to_cpu()
        fs2.to_cpu()
        self.check_equal_fs(self.fs, fs2)

    def check_copy_parameters_from(self, src_id, dst_id):
        aW = np.random.uniform(-1, 1, (2, 3)).astype(np.float32)
        ab = np.random.uniform(-1, 1, (2,)).astype(np.float32)
        bW = np.random.uniform(-1, 1, (2, 3)).astype(np.float32)
        bb = np.random.uniform(-1, 1, (2,)).astype(np.float32)
        params = [aW.copy(), ab.copy(), bW.copy(), bb.copy()]

        if dst_id >= 0:
            self.fs.to_gpu(dst_id)

        if src_id >= 0:
            params = [cuda.to_gpu(p, src_id) for p in params]

        self.fs.copy_parameters_from(params)
        self.fs.to_cpu()

        self.assertTrue((self.fs.a.params['W'].data == aW).all())
        self.assertTrue((self.fs.a.params['b'].data == ab).all())
        self.assertTrue((self.fs.b.params['W'].data == bW).all())
        self.assertTrue((self.fs.b.params['b'].data == bb).all())

    def test_copy_parameters_from_cpu_to_cpu(self):
        self.check_copy_parameters_from(-1, -1)

    @attr.gpu
    def test_copy_parameters_from_cpu_to_gpu(self):
        self.check_copy_parameters_from(-1, cuda.Device().id)

    @attr.gpu
    def test_copy_parameters_from_gpu_to_cpu(self):
        self.check_copy_parameters_from(cuda.Device().id, -1)

    @attr.gpu
    def test_copy_parameters_from_gpu_to_gpu(self):
        device_id = cuda.Device().id
        self.check_copy_parameters_from(device_id, device_id)

    @attr.multi_gpu(2)
    def test_copy_parameters_from_multigpu(self):
        self.check_copy_parameters_from(0, 1)

    def test_getitem(self):
        self.assertIs(self.fs['a'], self.fs.a)

    def test_getitem_notfoud(self):
        with self.assertRaises(KeyError):
            self.fs['not_found']


testing.run_module(__name__, __file__)
