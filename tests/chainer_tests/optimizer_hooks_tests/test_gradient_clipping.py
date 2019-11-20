import math
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


class SimpleLink(chainer.Link):

    def __init__(self, params):
        super(SimpleLink, self).__init__()
        with self.init_scope():
            for i, p in enumerate(params):
                setattr(self, 'p{}'.format(i), p)


@testing.backend.inject_backend_tests(None, _backend_params)
@testing.backend.inject_backend_tests(None, _backend_params)
@testing.backend.inject_backend_tests(None, _backend_params)
class TestGradientClipping(unittest.TestCase):

    def setUp(self):
        self.target = utils.ParametersLink.from_param_props(
            ((2, 3), (2, 0, 1), ()))
        self.norm = math.sqrt(sum([
            np.square(param.grad).sum() for param in self.target.params()]))

    def check_clipping(self, backend_configs, rate):
        target = self.target
        norm = self.norm
        assert len(backend_configs) == len(list(target.params()))
        devices = [bc.device for bc in backend_configs]

        threshold = norm * rate

        expects = []
        for param, device in zip(target.params(), devices):
            expects.append(param.array - param.grad * min(1, rate))
            param.to_device(device)

        opt = optimizers.SGD(lr=1)
        opt.setup(target)
        opt.add_hook(
            optimizer_hooks.GradientClipping(threshold))
        opt.update()

        for expect, param in zip(expects, target.params()):
            testing.assert_allclose(expect, param.array)

    def test_clipping(
            self, backend_config0, backend_config1, backend_config2):
        self.check_clipping(
            [backend_config0, backend_config1, backend_config2],
            0.5)

    def test_clipping_2(
            self, backend_config0, backend_config1, backend_config2):
        self.check_clipping(
            [backend_config0, backend_config1, backend_config2],
            2.0)


testing.run_module(__name__, __file__)
