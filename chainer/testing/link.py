import copy
import unittest

import numpy

import chainer
from chainer.testing import array as array_module
from chainer.testing import function as function_module


class LinkTestCase(unittest.TestCase):

    def generate_forward_backward_initializers(self):
        # Returns a tuple of initializers for forward and backward tests.
        #
        # The lengh of the tuple should be the same as the number of
        # parameters.
        raise NotImplementedError

    def generate_initializers(self):
        # Returns a tuple of lists, each list containing all initializers to be
        # tested for a particular parameter.
        #
        # The lengh of the tuple should be the same as the number of
        # parameters.
        raise NotImplementedError

    def create_link(self, initializers):
        # Returns a link object that is to be tested.
        raise NotImplementedError

    def generate_inputs(self):
        # Returns a tuple of input numpy.ndarrays to the ``__call__`` method.
        raise NotImplementedError

    def forward(self, link, inputs):
        # Default implementation of forward. This method may be overridden.
        #
        # assert all(isinstance(link, chainer.Link))
        # assert all(isinstance(a, chainer.Variable) for a in inputs)
        return link(*inputs)

    def forward_expected(self, inputs, params):
        # assert all(isinstance(a, numpy.ndarray) for a in inputs)
        # assert all(isinstance(p, numpy.ndarray) for p in params)
        raise NotImplementedError('forward_expected() is not implemented.')

    def on_sample(self):
        # Sets attributes of self that are to be used in the forward
        # definitions.
        # This method is called once in the backward test, followed by multiple
        # calls in case sampled points are invalid for the backward operation,
        # e.g. if the attribute is an array representing a nondifferentiable
        # point.
        pass

    def _generate_forward_backward_initializers(self, check=):
        initializers = self.generate_forward_backward_initializers()
        for i in initializers:
            assert (
                i is None
                or numpy.isscalar(i)
                or isinstance(i, chainer.Initializer)
                or isinstance(i, chainer.get_array_types()))
        return initializers

    def _generate_initializers(self):
        # TODO(hvy): Implement me.
        pass

    def _create_link(self, initializers):
        link = self.create_link(initializers)
        if not isinstance(link, chainer.Link):
            raise RuntimeError
        return link

    def _forward(self, inputs, backend_config):
        assert all(isinstance(a, chainer.Variable) for a in inputs)
        with backend_config:
            outputs = self.forward(inputs, backend_config.device)
        # TODO(hvy): Check outputs.
        return outputs

    def test_forward(self, backend_config):
        init_params = self.generate_init_params()
        link = self.create_link(init_params)

        # self.backend_config = backend_config

        inputs = self.generate_inputs()
        inputs_copied = [a.copy() for a in inputs]

        # TODO(hvy): Device transfers are carried out before this line.

        # After this call, ``__call__`` of the link is assumed to have been
        # called and parameters to have been initialized.
        outputs = self.forward(
            link, tuple([chainer.Variable(a) for a in inputs]))

        namedparams = link.namedparams()
        namedparams = dict(namedparams)
        for name in namedparams:
            param = namedparams[name]
            param.to_cpu()
            namedparams[name] = param.array
        expected_out = self.forward_expected(inputs_copied, namedparams)

    def test_backward(self, backend_config):
        # Checks that the graph is correct.
        pass
