import copy
import unittest


import chainer
from chainer.testing import array as array_module
from chainer.testing import function as function_module


class LinkTestCase(unittest.TestCase):

    def create_link(self, named_init_params):
        # Returns a link object to be tested.
        # assert all(
        #     isinstance(name, str) and isinstance(p, chainer.Parameter)
        #     for name, p in named_init_params.items()))
        raise NotImplementedError('create_link() is not implemented.')

    def generate_inputs(self):
        # Returns a tuple of input numpy.ndarrays to the ``__call__`` method.
        raise NotImplementedError('generate_inputs() is not implemented.')

    def forward(self, link, inputs):
        # assert all(isinstance(link, chainer.Link))
        # assert all(isinstance(a, chainer.Variable) for a in inputs)
        return link(*inputs)

    def forward_expected(self, inputs, params):
        # assert all(isinstance(a, numpy.ndarray) for a in inputs)
        # assert all(isinstance(p, numpy.ndarray) for p in params)
        raise NotImplementedError('forward_expected() is not implemented.')

    def generate_init_params(self):
        # Returns a dict of named params to the ``__init__`` method of the
        # link., i.e. {str: chainer.Parameter}.
        {}

    def on_sample(self):
        # Sets attributes of self that are to be used in the forward
        # definitions.
        # This method is called once in the backward test, followed by multiple
        # calls in case sampled points are invalid for the backward operation,
        # e.g. if the attribute is an array representing a nondifferentiable
        # point.
        pass

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
