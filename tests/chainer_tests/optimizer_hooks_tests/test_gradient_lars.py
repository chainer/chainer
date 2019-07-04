import unittest

import numpy as np

import chainer
from chainer import optimizer_hooks
from chainer import optimizers
from chainer import testing


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


class SimpleLink(chainer.Link):

    def __init__(self, params):
        super(SimpleLink, self).__init__()
        with self.init_scope():
            for i, p in enumerate(params):
                setattr(self, 'p{}'.format(i), p)


@testing.backend.inject_backend_tests(None, _backend_params)
@testing.backend.inject_backend_tests(None, _backend_params)
@testing.backend.inject_backend_tests(None, _backend_params)
class TestGradientLARS(unittest.TestCase):

    def setUp(self):
        num_params = 3
        arrs = [
            np.random.uniform(-3, 3, (2, 3)).astype(np.float32)
            for _ in range(num_params)]
        grads = [
            np.random.uniform(-3, 3, (2, 3)).astype(np.float32)
            for _ in range(num_params)]
        params_0 = []
        for arr, grad in zip(arrs, grads):
            param = chainer.Parameter(arr)
            param.grad = grad
            params_0.append(param)

        arrs = [
            np.random.uniform(-3, 3, (2, 3)).astype(np.float32) * 0.0001
            for _ in range(num_params)]
        grads = [
            np.random.uniform(-3, 3, (2, 3)).astype(np.float32)
            for _ in range(num_params)]
        params_1 = []
        for arr, grad in zip(arrs, grads):
            param = chainer.Parameter(arr)
            param.grad = grad
            params_1.append(param)

        self.target = chainer.ChainList(
            SimpleLink(params_0),
            SimpleLink(params_1))

    def check_LARS(self, backend_configs):
        target = self.target
        devices = [bc.device for bc in backend_configs]
        assert len(backend_configs) == len(list(target[0].params()))
        assert len(backend_configs) == len(list(target[1].params()))
        threshold = 1e-2
        weight_decay = 0.2
        eps = 1e-9

        expects0 = []
        expects1 = []
        # Compute expected
        for param, device in zip(target[0].params(), devices):
            p0_norm = np.linalg.norm(param.array)
            g0_norm = np.linalg.norm(param.grad)
            clip_rate = p0_norm / (eps + g0_norm + weight_decay * p0_norm)
            expects0.append(param.array - clip_rate
                            * (param.grad + weight_decay * param.array))
            param.to_device(device)

        for param, device in zip(target[1].params(), devices):
            expects1.append(param.array - 1.0
                            * (param.grad + weight_decay * param.array))

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(optimizer_hooks.GradientLARS(threshold=threshold,
                                                  weight_decay=weight_decay,
                                                  eps=eps))
        opt.update()
        for expect, param in zip(expects0, target[0].params()):
            testing.assert_allclose(expect, param.array)
        for expect, param in zip(expects1, target[1].params()):
            testing.assert_allclose(expect, param.array)

    def test_LARS(self, backend_config0,
                  backend_config1, backend_config2):
        self.check_LARS(
            [backend_config0, backend_config1, backend_config2])


testing.run_module(__name__, __file__)
