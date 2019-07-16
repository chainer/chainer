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
class TestGradientHardClipping(unittest.TestCase):

    def setUp(self):
        num_params = 3
        arrs = [
            np.random.uniform(-3, 3, (2, 3)).astype(np.float32)
            for _ in range(num_params)]
        grads = [
            np.random.uniform(-3, 3, (2, 3)).astype(np.float32)
            for _ in range(num_params)]
        params = []
        for arr, grad in zip(arrs, grads):
            param = chainer.Parameter(arr)
            param.grad = grad
            params.append(param)

        self.target = SimpleLink(params)

    def check_hardclipping(self, backend_configs):
        target = self.target
        assert len(backend_configs) == len(list(target.params()))
        devices = [bc.device for bc in backend_configs]

        lower_bound = -0.9
        upper_bound = 1.1
        expects = []
        # Compute expected
        for param, device in zip(target.params(), devices):
            expects.append(param.array - np.clip(param.grad,
                                                 lower_bound, upper_bound))
            param.to_device(device)

        # Apply optimizer_hook
        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(
            optimizer_hooks.GradientHardClipping(lower_bound, upper_bound))
        opt.update()

        # Validate
        for expect, param in zip(expects, target.params()):
            testing.assert_allclose(expect, param.array)

    def test_hardclipping(self, backend_config0,
                          backend_config1, backend_config2):
        self.check_hardclipping(
            [backend_config0, backend_config1, backend_config2])


testing.run_module(__name__, __file__)
