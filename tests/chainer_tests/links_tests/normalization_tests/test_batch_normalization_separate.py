import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


# Alt. 2: Using a base class to define input generation and forward
# propagation.
#
# - Maybe we cannot generate inputs without the parameterizations.
#   - Actually, we can! With a @property of e.g. dtype that raises by defualt!
# - Do we need parameterization? Yes for forward/backward, but not for
# initializers.


# TODO(hvy): This class derives from object. Maybe we should provide a base
# class for this class that defines e.g. properties for dtype and shape
# that throws?
# TODO(hvy): Test multiple backends.
class BatchNormalizationLinkTest(LinkTestCaseBase):
    """
    Common logic for forward/backward and initialization tests.
    """

    required_attrs = ('ndim', 'axis', 'test', 'dtype')

    def setUp(self):
        self.finetune = False

        if not hasattr(self, 'axis'):
            aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))
            shape = (5, 3) + (2,) * self.ndim
            param_shape = shape[1]
            self.expander = (None, Ellipsis) + (None,) * self.ndim
        else:
            aggr_axes = self.axis
            if isinstance(self.axis, int):
                aggr_axes = self.axis,
            shape = self.input_shape
            param_shape = tuple(
                s
                for i, s in enumerate(shape)
                if i not in aggr_axes
            )
            self.expander = tuple(
                None if i in aggr_axes else slice(None)
                for i in range(len(shape))
            )
        self.shape = shape
        self.param_shape = shape


    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def create_link(self, initializers):
        initial_gamma, initial_beta = initializers

        initial_avg_mean_var = {}
        if self.test:
            initial_avg_mean_var['initial_avg_mean'] = numpy.random.uniform(
                -1, 1, self.param_shape).astype(self.dtype)
            initial_avg_mean_var['initial_avg_var'] = numpy.random.uniform(
                0.5, 1, self.param_shape).astype(self.dtype)

        link = links.BatchNormalization(
            axis=axis,
            initial_gamma=initial_gamma,
            initial_beta=initial_beta,
            **initial_avg_mean_var)

        return link

    def forward(self, link, inputs):
        assert isinstance(input[0], chainer.Variable)

        x, = inputs
        y = link(x, finetune=self.finetune)
        return y


_inject_backedn_tests = (
    testing.inject_backend_tests(
        None,
        # CPU tests
        [
            {},
            {'use_ideep': 'always'},
        ]
        # GPU tests
        + testing.product({
            'use_cuda': [True],
            'use_cudnn': ['never', 'always'],
            'cuda_device': [0, 1],
        })
        # ChainerX tests
        + [
            {'use_chainerx': True, 'chainerx_device': 'native:0'},
            {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
            {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
        ]))


@testing.parameterize(*(testing.product_dict(
    testing.product({
        'test': [True, False],
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
    }),
    testing.product({
        'ndim': [0, 1, 2, 3],
    }) + [
        {'input_shape': (5, 4, 3, 2), 'axis': (0, 2, 3)},
        {'input_shape': (5, 4), 'axis': 0},
        {'input_shape': (5, 4, 3), 'axis': (0, 1)},
    ]
)))
@_inject_backedn_tests
class BatchNormalizationForwardBackwardTest(
        BatchNormalizationLinkTest, LinkTestCase):

    # Passed to forward_expect as numpy.ndarrays.
    params = ['gamma', 'beta']

    # Passed to forward_expect as numpy.ndarrays.
    persistent_values = ['avg_mean', 'avg_var']

    # Passed to forward_expect as is.
    attributes = ['eps']

    def forward_expected(self, inputs, params, persistent_values, attributes):
        assert isinstance(inputs[0], numpy.ndarray)
        assert all(isinstance(p, numpy.ndarray) p for p in params)
        assert all(isinstance(p, numpy.ndarray) p for p in persistent_values)

        x, = inputs
        gamma, beta = params
        mean, var = persistent_values
        eps, = attributes

        mean = avg_mean[self.expander]
        var = avg_var[self.expander]

        if self.test:
            std = numpy.sqrt(var)
        else:
            std = numpy.sqrt(var + eps)
        y = gamma * (x - mean) / std + beta
        return y


@testing.parameterize({
    'initial_gamma': [I.Constant(2), 1, None],
    'initial_beta': [I.Constant(2), 1, None],
})
@_inject_backedn_tests
class BatchNormalizationInitializersTest(
        BatchNormalizationLinkTest, LinkInitializerTestCase):

    ndim = 2
    axis = (0, 2, 3)
    test = False
    dtype = numpy.float32
