import unittest

import numpy as np

import chainer
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
class TestGradientLARS(unittest.TestCase):

    def setUp(self):
        link1 = utils.ParametersLink.from_param_props(
            ((2, 3), (2, 0, 1), (0,)))
        link2 = utils.ParametersLink.from_param_props(
            ((5, 0, 1), (0,), (7, 3)))
        for param in link2.params():
            param.array[...] *= 0.0001
        self.target = chainer.ChainList(link1, link2)

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
