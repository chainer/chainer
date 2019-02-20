import copy
import itertools
import unittest

import numpy

import chainer
from chainer import backend
from chainer import initializers
from chainer.testing import array as array_module
from chainer.testing import function as function_module


class ConvertedInitializer(object):

    def __init__(self, init_from, init_to):
        self.init_from = init_from
        self.init_to = init_to


def _check_initializers(inits):
    for init in inits:
        if isinstance(init, chainer.Initializer):
            continue
        elif isinstance(init, chainer.get_array_types()):
            continue
        elif numpy.isscalar(init):
            # TODO(hvy): Check if the above condition is correct w.r.t. how
            # link constructors interpret scalaras and NumPy scalars.
            continue
        elif isinstance(init, ConvertedInitializer):
            continue
        raise ValueError('Initializer is invalid')


class LinkTestCase(unittest.TestCase):

    # List of parameter keys represented as strings.
    # E.g. ['gamma', 'beta'] for BatchNormalization.
    param_names = []

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

    def _generate_forward_backward_initializers(self):
        initializers = self.generate_forward_backward_initializers()
        for i in initializers:
            assert (
                i is None
                or numpy.isscalar(i)
                or isinstance(i, chainer.Initializer)
                or isinstance(i, chainer.get_array_types()))
        return initializers

    def _generate_initializers(self):
        params_inits = self.generate_initializers()
        for param_inits in params_inits:
            _check_initializers(param_inits)
        return params_inits

    def _generate_inputs(self):
        inputs = self.generate_inputs()
        _check_array_types(inputs, backend.CpuDevice(), 'generate_inputs')
        return inputs

    def _create_link(self, initializers):
        link = self.create_link(initializers)
        if not isinstance(link, chainer.Link):
            raise RuntimeError
        return link

    def _forward(self, link, inputs):
        assert all(isinstance(a, chainer.Variable) for a in inputs)
        outputs = self.forward(link, inputs)
        # TODO(hvy): Check outputs.
        return outputs

    def _create_link(self, initializers, device):
        link = self.create_link(initializers)
        if not isinstance(link, chainer.Link):
            raise ValueError('`create_link` must return a chainer.Link object.')
        link.to_device(device)
        return link

    def _test_forward(self, inits, backend_config):
        assert all(i is not None for i in inits)

        inits_processed = []
        for init in inits:
            if isinstance(init, ConvertedInitializer):
                init = init.init_from
            inits_processed.append(init)

        device = backend_config.device

        link = self._create_link(inits_processed, device)

        inputs_np = self._generate_inputs()
        inputs_xp = tuple([device.send(i) for i in inputs_np])
        input_vars = tuple([chainer.Variable(i) for i in inputs_xp])

        cpu_device = backend.CpuDevice()

        with backend_config:
            actual_outputs_var = self._forward(link, input_vars)

        actual_outputs_xp = [v.array for v in actual_outputs_var]
        actual_outputs_np = [cpu_device.send(arr) for arr in actual_outputs_xp]

        params = []
        for param_name in self.param_names:
            param = getattr(link, param_name, None)
            if param is None:
                raise RuntimeError
            params.append(param)

        params_np = []
        for param in params:
            param_xp = param.array
            param_np = cpu_device.send(param_xp)
            params_np.append(param_np)


        expected_outputs_np = self.forward_expected(inputs_np, params_np)

        # print(actual_outputs_np - expected_outputs_np)
        opts = {}
        for e, a in zip(expected_outputs_np, actual_outputs_np):
            array_module.assert_allclose(e, a, **self.check_forward_options)


    def test_forward(self, backend_config):
        params_inits = self._generate_initializers()
        for params_init in itertools.zip_longest(*params_inits):
            self._test_forward(params_init, backend_config)

        return

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


def _check_array_types(arrays, device, func_name):
    if not isinstance(arrays, tuple):
        raise TypeError(
            '`{}()` must return a tuple, '
            'not {}.'.format(func_name, type(arrays)))
    if not all(isinstance(a, device.supported_array_types) for a in arrays):
        raise TypeError(
            '{}() must return a tuple of arrays supported by device {}.\n'
            'Actual: {}'.format(
                func_name, device, tuple([type(a) for a in arrays])))
