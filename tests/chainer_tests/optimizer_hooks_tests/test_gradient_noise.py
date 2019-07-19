import unittest
import itertools

import mock
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
class TestGradientNoise(unittest.TestCase):

    eta = 0.01

    def setUp(self):
        self.target = utils.ParametersLink.from_param_props(
            # TODO(niboshi): Use different shapes
            ((2, 3), (2, 3), (2, 3)))

        self.noise_value = np.random.normal(
            loc=0, scale=np.sqrt(self.eta / np.power(1, 0.55)),
            size=(2, 3)).astype(np.float32)

    def check_gradient_noise(self, backend_configs):
        target = self.target
        assert len(backend_configs) == len(list(target.params()))
        devices = [bc.device for bc in backend_configs]
        noise_value = np.asarray(self.noise_value)
        expects = []
        # Compute expected
        for param, device in zip(target.params(), devices):
            expects.append(param.array - param.grad - noise_value)
            param.to_device(device)

        def test_noise(xp, shape, dtype, hook, opt):
            # Make noise value an array of current backend
            return xp.array(noise_value)

        noise = mock.Mock(side_effect=test_noise)
        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        hook = optimizer_hooks.GradientNoise(self.eta, noise_func=noise)
        opt.add_hook(hook)
        opt.update()

        # Validate
        for expect, param in zip(expects, target.params()):
            testing.assert_allclose(expect, param.array)

        self.assertEqual(noise.call_count, len(tuple(self.target.params())))

        calls = []
        for param in target.params():
            xp = param.device.xp
            calls.append(mock.call(xp, (2, 3), np.dtype('float32'), hook,
                                   param.update_rule))

        # Order does not matter
        assert(any([noise.mock_calls == list(permuted_calls)
                    for permuted_calls in itertools.permutations(calls)]))

    def test_gradient_noise(self, backend_config0,
                            backend_config1, backend_config2):
        self.check_gradient_noise(
            [backend_config0, backend_config1, backend_config2])


testing.run_module(__name__, __file__)
