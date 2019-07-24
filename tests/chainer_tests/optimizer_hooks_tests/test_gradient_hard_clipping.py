import unittest

import numpy as np

from chainer import optimizer_hooks
from chainer import optimizers
from chainer import testing

import utils


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


@testing.backend.inject_backend_tests(None, _backend_params)
@testing.backend.inject_backend_tests(None, _backend_params)
@testing.backend.inject_backend_tests(None, _backend_params)
class TestGradientHardClipping(unittest.TestCase):

    def setUp(self):
        self.target = utils.ParametersLink.from_param_props(
            ((2, 3), (2, 0, 1), ()))

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
