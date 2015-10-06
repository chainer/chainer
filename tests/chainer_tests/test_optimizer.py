import unittest

import mock
import numpy as np

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import optimizer
from chainer import optimizers
from chainer import testing
from chainer.testing import attr


class TestOptimizerWeightDecay(unittest.TestCase):

    def setUp(self):
        self.w = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.g = np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3)

    def check_weight_decay(self, w, g):
        decay = 0.2
        expect = w - g - decay * w

        opt = optimizers.SGD(lr=1)
        opt.setup((w, g))
        opt.weight_decay(decay)
        opt.update()

        gradient_check.assert_allclose(expect, w)

    def test_weight_decay_cpu(self):
        self.check_weight_decay(self.w, self.g)

    @attr.gpu
    def test_weight_decay_gpu(self):
        self.check_weight_decay(cuda.to_gpu(self.w), cuda.to_gpu(self.g))


class TestGradientMethod(unittest.TestCase):

    def _suffix(self, gpu):
        if gpu:
            return 'gpu'
        else:
            return 'cpu'

    def _get_method(self, prefix, gpu):
        return getattr(self.optimizer, prefix + '_' + self._suffix(gpu))

    def setUp(self):
        opt = chainer.GradientMethod()
        opt.init_state_cpu = mock.MagicMock()
        opt.init_state_gpu = mock.MagicMock()
        opt.update_param_cpu = mock.MagicMock()
        opt.update_param_gpu = mock.MagicMock()
        self.optimizer = opt

        self.params = [np.arange(3).astype(np.float32)]
        self.grads = [np.arange(3).astype(np.float32)]

    def setup_cpu(self):
        self.optimizer.setup((self.params, self.grads))

    def setup_gpu(self, dst_id=None):
        self.params = [cuda.to_gpu(p, dst_id) for p in self.params]
        self.grads = [cuda.to_gpu(p, dst_id) for p in self.grads]
        self.optimizer.setup((self.params, self.grads))

    def check_init_state(self, param, grad, gpu):
        state = self.optimizer.init_state(param, grad)

        self._get_method('init_state', gpu).assert_called_once_with(
            param, grad)
        self.assertEqual(self._get_method('init_state', not gpu).call_count, 0)

    def test_init_state_cpu(self):
        param = np.arange(3)
        grad = np.arange(3)
        self.check_init_state(param, grad, False)

    @attr.gpu
    def test_init_state_gpu(self):
        param = cuda.to_gpu(np.arange(3))
        grad = cuda.to_gpu(np.arange(3))
        self.check_init_state(param, grad, True)

    def check_update(self, gpu):
        self.assertEqual(self.optimizer.t, 0)

        self.optimizer.update()
        self.assertEqual(self.optimizer.t, 1)

        self._get_method('update_param', gpu).assert_called_once_with(
            self.params[0], self.grads[0], {})
        self.assertEqual(self._get_method('update_param', not gpu).call_count, 0)

        self.optimizer.zero_grads()
        self.assertTrue((cuda.to_cpu(self.grads[0]) == 0).all())

    def test_update_cpu(self):
        self.setup_cpu()
        self.check_update(False)

    @attr.gpu
    def test_update_gpu(self):
        self.setup_gpu()
        self.check_update(True)

    def check_accumulate_grads_from_cpu(self):
        self.optimizer.accumulate_grads([np.arange(3)])
        self.assertTrue((cuda.to_cpu(self.grads[0]) == np.arange(3) * 2).all())

    @attr.gpu
    def check_accumulate_grads_from_gpu(self, src_id):
        with cuda.Device(src_id):
            self.optimizer.accumulate_grads([cuda.cupy.arange(3)])
        self.assertTrue((cuda.to_cpu(self.grads[0]) == np.arange(3) * 2).all())

    def test_accumulate_grads_cpu_to_cpu(self):
        self.setup_cpu()
        self.check_accumulate_grads_from_cpu()

    @attr.gpu
    def test_accumulate_grads_cpu_to_gpu(self):
        self.setup_gpu()
        self.check_accumulate_grads_from_cpu()

    @attr.gpu
    def test_accumulate_grads_gpu_to_cpu(self):
        self.setup_cpu()
        self.check_accumulate_grads_from_gpu(cuda.Device().id)

    @attr.gpu
    def test_accumulate_grads_gpu_to_gpu(self):
        device_id = cuda.Device().id
        self.setup_gpu(device_id)
        self.check_accumulate_grads_from_gpu(device_id)

    @attr.multi_gpu(2)
    def test_accumulate_grads_multigpu(self):
        self.setup_gpu(0)
        self.check_accumulate_grads_from_gpu(1)

    def check_compute_grads_norm(self):
        norm = self.optimizer.compute_grads_norm()
        self.assertAlmostEqual(norm, np.sqrt(5))

    def test_compute_grads_norm_cpu(self):
        self.setup_cpu()
        self.check_compute_grads_norm()

    @attr.gpu
    def test_compute_grads_norm_gpu(self):
        self.setup_gpu()
        self.check_compute_grads_norm()

    def check_weight_decay(self):
        self.optimizer.weight_decay(0.1)
        g = cuda.to_cpu(self.grads[0])
        expect = np.array([0.0, 1.1, 2.2], dtype=np.float32)
        gradient_check.assert_allclose(g, expect)

    def test_weight_decay_cpu(self):
        self.setup_cpu()
        self.check_weight_decay()

    @attr.gpu
    def test_weight_decay_gpu(self):
        self.setup_gpu()
        self.check_weight_decay()

    def check_clip_grads(self):
        self.optimizer.clip_grads(1.0)
        g = cuda.to_cpu(self.grads[0])
        sqnorm = g.dot(g)
        self.assertAlmostEqual(sqnorm, 1.0, delta=1.0e-5)

    def test_clip_grads_cpu(self):
        self.setup_cpu()
        self.check_clip_grads()

    @attr.gpu
    def test_clip_grads_gpu(self):
        self.setup_gpu()
        self.check_clip_grads()


testing.run_module(__name__, __file__)
