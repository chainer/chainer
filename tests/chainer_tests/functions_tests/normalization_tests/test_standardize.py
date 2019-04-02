import functools
import unittest

import numpy

from chainer.functions.normalization._standardize import _standardize
from chainer import testing


def _skip_if(cond, reason):
    """Skip test if cond(self) is True"""
    def decorator(impl):
        @functools.wraps(impl)
        def wrapper(self, *args, **kwargs):
            if cond(self):
                raise unittest.SkipTest(reason)
            else:
                impl(self, *args, **kwargs)
        return wrapper
    return decorator


def _is_good_param(param):
    # Check if 'nonzero' param is valid and meaningful. On the latter point,
    # x should contain at least a zero if 'nonzeros' param is given.
    return param['nonzeros'] is None \
        or param['nonzeros'] < numpy.prod(param['shape'])


@testing.parameterize(*filter(
    _is_good_param,
    testing.product({
        'ch_out': [1, 5],
        'size': [10, 20],
        'dtype': [numpy.float32, numpy.float16],
    })
    + testing.product([
        # nonzeros (optional int): max number of nonzero elems in input
        # truezero (bool): flag whether zero elems are exactly zero. If false,
        #     randomly-chosen small values are used.
        {'eps': 1e-5, 'nonzeros': None},
        {'eps': 1e-1, 'nonzeros': None},
        {'eps': 1e-1, 'nonzeros': 0, 'truezero': True},
        {'eps': 1e-1, 'nonzeros': 0, 'truezero': False},
        {'eps': 1e-1, 'nonzeros': 2, 'truezero': True},
        {'eps': 1e-1, 'nonzeros': 2, 'truezero': False},
    ])
))
@testing.backend.inject_backend_tests(
    None,
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)
class TestStandardize(testing.FunctionTestCase):

    def setUp(self):
        self.skip_double_backward_test = (self.nonzeros is not None)
        if self.nonzeros is not None:
            # Make self.x have limited number of large values

            # get mask of indices to modify at
            zeros = self.x.size - self.nonzeros
            while True:
                rand = numpy.random.uniform(0, 1, self.shape)
                mask = rand <= numpy.sort(rand.ravel())[zeros - 1]
                if self.x[mask].shape == (zeros,):
                    break

            # set zeros or small values to a part of the input
            if self.truezero:
                self.x[mask] = 0
            else:
                zero_scale = 10. ** numpy.random.randint(-40, -3)
                self.x[mask] = numpy.random.uniform(
                    -zero_scale, zero_scale, zeros)
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 5e-3, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 5e-3, 'rtol': 1e-2})
            self.check_double_backward_options.update(
                {'atol': 5e-3, 'rtol': 1e-2})

    def generate_inputs(self):
        shape = self.ch_out, self.size
        x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return _standardize(x, self.eps),

    def forward_expected(self, inputs):
        x, = inputs
        mu = numpy.mean(x, axis=1, keepdims=True)
        x_mu = x - mu
        var = numpy.mean(numpy.square(x_mu), axis=1, keepdims=True)
        std = numpy.sqrt(var, dtype=x.dtype) + self.eps
        inv_std = 1. / std
        return x_mu * inv_std,


testing.run_module(__name__, __file__)
