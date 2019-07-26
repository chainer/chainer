import unittest

import numpy as np

import chainer
import chainer.functions as F
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
class TestWeightDecay(unittest.TestCase):

    def setUp(self):
        self.target = utils.ParametersLink.from_param_props(
            ((2, 3), (2, 0, 1), ()))

    def check_weight_decay(self, backend_configs):
        target = self.target
        assert len(backend_configs) == len(list(target.params()))
        devices = [bc.device for bc in backend_configs]

        decay = 0.2

        # Compute expected
        expects = []
        for param, device in zip(target.params(), devices):
            expects.append(param.array - param.grad - decay * param.array)
            param.to_device(device)

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(optimizer_hooks.WeightDecay(decay))
        opt.update()

        # Validate
        for expect, param in zip(expects, target.params()):
            testing.assert_allclose(expect, param.array)

    def test_weight_decay(self, backend_config0,
                          backend_config1, backend_config2):
        self.check_weight_decay(
            [backend_config0, backend_config1, backend_config2])


# TODO(kshitij12345): Test with chainerx when `loss_scale` in `backward`.
@testing.inject_backend_tests(
    None,
    # CPU tests
    [{}, {'use_ideep': 'always'}]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
    })
)
class TestWeightDecayLossScale(unittest.TestCase):

    def test_weight_decay_loss_scale(self, backend_config):
        a = self._updated_array(backend_config, None)
        b = self._updated_array(backend_config, loss_scale=4.)
        testing.assert_allclose(a, b)

    def _updated_array(self, backend_config, loss_scale):
        arr = np.arange(3, dtype=np.float32)
        param = chainer.Parameter(arr)
        link = chainer.Link()
        with link.init_scope():
            link.p = param
        link.to_device(backend_config.device)
        opt = optimizers.SGD(lr=1)
        opt.setup(link)
        opt.add_hook(optimizer_hooks.WeightDecay(1/8.))
        loss = F.sum(link.p ** 3)
        loss.backward(loss_scale=loss_scale)
        opt.update()
        return link.p.array


testing.run_module(__name__, __file__)
