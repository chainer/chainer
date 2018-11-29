import chainer
import chainer.testing
import chainer.testing.attr
import chainermn
import mock
import numpy as np
import unittest


class ExampleModel(chainer.Chain):
    def __init__(self):
        super(ExampleModel, self).__init__()
        with self.init_scope():
            self.a = chainer.links.Linear(2, 3)
            self.b = chainer.links.Linear(3, 4)
            self.c = chainer.links.Linear(4, 5)


class TestMultiNodeOptimizer(unittest.TestCase):


    def test_get_memory_size(self):
        self.xp = np
        self.actual_optimizer = chainer.GradientMethod()
        self.comm = chainermn.create_communicator('naive')
        self.optimizer = chainermn.create_multi_node_optimizer(
            self.actual_optimizer, self.comm)

        dtype = self.xp.dtype(self.xp.float32)
        param = chainer.Variable(
                    data=self.xp.array([self.comm.rank for _ in range(3)], dtype=dtype),
            grad=self.xp.array([self.comm.rank for _ in range(3)], dtype=dtype))

        memory_size = self.optimizer.get_memory_size(param, dtype)
        self.assertEqual(memory_size, 12)

"""
    def setup_cpu(self):
        self.comm = chainermn.create_communicator('naive')
        self.target = ExampleModel()
        self.target.a.W.data[:] = self.comm.rank
        self.target.b.W.data[:] = self.comm.rank + 1
        self.target.c.W.data[:] = self.comm.rank + 2
        self.target.a.W.grad[:] = 0
        self.target.b.W.grad[:] = 0
        self.target.c.W.grad[:] = 0
        self.actual_optimizer = chainer.GradientMethod()
        self.actual_optimizer.create_update_rule = mock.MagicMock

    def setup_gpu(self, device=None):
        self.comm = chainermn.create_communicator('hierarchical')
        device = self.comm.intra_rank
        chainer.cuda.get_device_from_id(device).use()
        self.target = ExampleModel()
        self.target.to_gpu()
        self.target.a.W.data[:] = self.comm.rank
        self.target.b.W.data[:] = self.comm.rank + 1
        self.target.c.W.data[:] = self.comm.rank + 2
        self.target.a.W.grad[:] = 0
        self.target.b.W.grad[:] = 0
        self.target.c.W.grad[:] = 0
        self.actual_optimizer = chainer.GradientMethod()
        self.actual_optimizer.create_update_rule = mock.MagicMock

    def test_update_with_cpu(self):
        self.setup_cpu()
        self.optimizer = chainermn.create_multi_node_optimizer(
            self.actual_optimizer, self.comm)
        opt = self.optimizer.setup(self.target)
        assert opt is self.optimizer
        self.optimizer.update()
        self.assertEqual(self.actual_optimizer.t, 0)
        self.optimizer.target.a.W.grad[:] = self.comm.rank
        self.optimizer.target.b.W.grad[:] = self.comm.rank + 1
        self.optimizer.target.c.W.grad[:] = self.comm.rank + 2

        self.optimizer.update()
        self.assertEqual(self.actual_optimizer.t, 1)
        self.optimizer.target.a.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.a.W)
        self.optimizer.target.b.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.b.W)
        self.optimizer.target.c.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.c.W)

        base = (self.comm.size - 1.0) / 2
        chainer.testing.assert_allclose(self.optimizer.target.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(self.optimizer.target.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(self.optimizer.target.c.W.grad,
                                        (base + 2) * np.ones((5, 4)))

    @chainer.testing.attr.gpu
    def test_update_with_gpu(self):
        self.setup_gpu()
        self.optimizer = chainermn.create_multi_node_optimizer(
            self.actual_optimizer, self.comm)
        opt = self.optimizer.setup(self.target)
        assert opt is self.optimizer
        self.optimizer.update()
        self.assertEqual(self.actual_optimizer.t, 0)
        self.optimizer.target.a.W.grad[:] = self.comm.rank
        self.optimizer.target.b.W.grad[:] = self.comm.rank + 1
        self.optimizer.target.c.W.grad[:] = self.comm.rank + 2

        self.optimizer.update()
        self.assertEqual(self.actual_optimizer.t, 1)
        self.optimizer.target.a.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.a.W)
        self.optimizer.target.b.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.b.W)
        self.optimizer.target.c.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.c.W)

        base = (self.comm.size - 1.0) / 2
        chainer.testing.assert_allclose(self.optimizer.target.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(self.optimizer.target.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(self.optimizer.target.c.W.grad,
                                        (base + 2) * np.ones((5, 4)))
"""
