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

# Alt. 1: Using a single class to test both forward/backward and initializers.
#
# Two steps:
#
# 1. Run forward, backward tests with standard parameterizations and injected
# backend.
#
# 2. Run initializer tests for each initializer.
#    - But those will be parameterized too. Can we avoid it?


# TODO(hvy): Test multiple backends.
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
class BatchNormalizationTest(LinkTestCase):

    # Belows are "keys" that are used to extract information for
    # forward_expect. Without where, we need to pass the link object to
    # forwad_expect but we don't want to do that?
    #
    # The benefit of declaring them here is that they are
    # readable, and also allows the base class to do some
    # "test_params_registerd", etc. to check that parameters and persistent
    # varlues are actually registered correctly in the link.

    # Passed to forward_expect as numpy.ndarrays.
    params = ['gamma', 'beta']

    # Passed to forward_expect as numpy.ndarrays.
    persistent_values = ['avg_mean', 'avg_var']

    # Passed to forward_expect as is.
    attributes = ['eps']


    def setUp(self):
        # Sets the follwing test attributes.
        # self.finetune
        # self.expander
        # self.shape
        # self.param_shape

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

    def generate_initializers(self):
        # Forward and backward tests should use random ndarray-initialized
        # parameters.
        initial_gamma = numppy.random.uniform(
            -1, 1, self.axis).astype(self.dtype),
        initial_beta = numpy.random.uniform(
            -1, 1, self.axis).astype(self.dtype)
        # return {'initial_gamma': initial_gamma, 'initial_beta': initial_beta}
        return initial_gamma, initial_beta

    @property
    def initializers(self):
        # Various initializers are supported by BN and these should be tested.
        initial_gamma = [I.Constant(2), 1, None]
        initial_beta = [I.Constant(2), 1, None]
        # return {'initial_gamma': initial_gamma, 'initial_beta': initial_beta}
        return initial_gamma, initial_beta

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

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, link, inputs):
        assert isinstance(input[0], chainer.Variable)

        x, = inputs
        y = link(x, finetune=self.finetune)
        return y

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

