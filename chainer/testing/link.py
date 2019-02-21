import itertools
import sys
import unittest

import numpy

import chainer
from chainer import backend
from chainer import gradient_check
from chainer import initializers
from chainer.testing import array as array_module


class InitializerPair(object):

    """Class to hold a pair of initializer-like objects.

    The initializer-like objects can be accessed via the ``first`` and
    ``second`` attributes.

    When implementing ``LinkTestCase``, instances of this class can be included
    in lists enumerating all initializer-like objects that should be tested.
    In that case, the first element should correspond to the initializer-like
    argument passed to the constructor of the ``Link``, and the second element
    correspond to the expected initializer-like object upon parameter
    initialization.
    """

    def __init__(self, first, second):
        initializers._check_is_initializer_like(first)
        initializers._check_is_initializer_like(second)

        self.first = first
        self.second = second


class LinkTestCase(unittest.TestCase):

    backend_config = None
    check_forward_options = {}
    check_backward_options = {}
    skip_forward_test = False
    skip_backward_test = False
    skip_initializers_test = False
    dodge_nondifferentiable = False
    contiguous = None

    # List of parameter names represented as strings.
    # I.e. ['gamma', 'beta'] for BatchNormalization.
    param_names = []

    _default_initializer = None

    @property
    def default_initializer(self):
        return self._default_initializer

    @default_initializer.setter
    def default_initializer(self, default_initializer):
        initializers._check_is_initializer_like(default_initializer)
        self._default_initializer = default_initializer

    def generate_initializers(self):
        """Returns initializers.

        Returns:
            A tuple of lists, each list containing all initializers to be
            tested for a particular parameter.
            The lengh of the tuple should be the same as the number of
            parameters.
        """
        raise NotImplementedError('generate_initializers is not implemented.')

    def create_link(self, initializers):
        """Returns a link object that is to be tested."""
        raise NotImplementedError('create_link is not implemented.')

    def generate_inputs(self):
        """Returns a tuple of input numpy.ndarrays to the ``__call__`` method.
        """
        raise NotImplementedError('generate_inputs is not implemented.')

    def forward(self, link, inputs):
        """Computes and returns the result of a forward pass."""
        # assert all(isinstance(link, chainer.Link))
        # assert all(isinstance(a, chainer.Variable) for a in inputs)
        return link(*inputs)

    def forward_expected(self, inputs, params):
        """Returns the expected results of a forward pass."""
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

    def generate_grad_outputs(self, outputs_template):
        """Returns upstream gradients."""
        grad_outputs = tuple([
            numpy.random.uniform(-1, 1, a.shape).astype(a.dtype)
            for a in outputs_template])
        return grad_outputs

    def test_forward(self, backend_config):
        if self.skip_forward_test:
            raise unittest.SkipTest('skip_forward_test is set')

        params_inits = self._generate_initializers()
        for params_init in _initializer_iterator(
                params_inits, fill_init=self.default_initializer):
            self._test_single_forward(params_init, backend_config)

    def test_backward(self, backend_config):
        # Checks that the graph is correct.
        if self.skip_backward_test:
            raise unittest.SkipTest('skip_backward_test is set')

        params_inits = self._generate_initializers()

        for params_init in _initializer_iterator(
                params_inits, fill_init=self.default_initializer):
            self._test_single_backward(params_init, backend_config)

    def test_initializers(self, backend_config):
        """Tests that the parameters of a links are correctly initialized."""
        if self.skip_initializers_test:
            raise unittest.SkipTest('skip_initializers_test is set')

        params_inits = self._generate_initializers()

        for params_init in _initializer_iterator(
                params_inits, fill_init=self.default_initializer):
            self._test_single_initializers(params_init, backend_config)

    def _generate_initialized_link(self, inits, backend_config):
        inits = _get_initializers(inits)
        device = backend_config.device
        link = self._create_link(inits, device)

        # Generate inputs and compute a forward pass to initialize the
        # parameters.
        inputs_np = self._generate_inputs()
        inputs_xp = [device.send(i) for i in inputs_np]
        input_vars = [chainer.Variable(i) for i in inputs_xp]
        outputs_var = self._forward(link, input_vars, backend_config)
        cpu_device = backend.CpuDevice()
        outputs_xp = [v.array for v in outputs_var]
        outputs_np = [cpu_device.send(arr) for arr in outputs_xp]

        return link, inputs_np, outputs_np

    # TODO(hvy): Make this a free function.
    def _get_param_arrays(self, link):
        cpu_device = backend.CpuDevice()

        params = []
        for param_name in self.param_names:
            param = getattr(link, param_name, None)
            if param is None:
                raise RuntimeError(
                    'Link does not have a parameter named \'{}\'.'.format(
                        param_name))
            params.append(param)

        params_np = []
        for param in params:
            param_xp = param.array
            param_np = cpu_device.send(param_xp)
            params_np.append(param_np)

        return params_np

    def _generate_initializers(self):
        params_inits = self.generate_initializers()
        _check_generated_initializers(params_inits)
        return params_inits

    def _create_link(self, initializers, device):
        link = self.create_link(initializers)
        if not isinstance(link, chainer.Link):
            raise ValueError(
                '`create_link` must return a chainer.Link object.')
        link.to_device(device)
        return link

    def _generate_inputs(self):
        inputs = self.generate_inputs()
        _check_array_types(inputs, backend.CpuDevice(), 'generate_inputs')
        return inputs

    def _forward(self, link, inputs, backend_config):
        assert all(isinstance(x, chainer.Variable) for x in inputs)
        with backend_config:
            outputs = self.forward(link, inputs)
        # TODO(hvy): Check outputs.
        return outputs

    def _forward_expected(self, inputs, params):
        assert all(isinstance(x, numpy.ndarray) for x in inputs)
        assert all(isinstance(x, numpy.ndarray) for x in params)

        outputs = self.forward_expected(inputs, params)
        return outputs

    def _generate_grad_outputs(self, outputs_template):
        grad_outputs = self.generate_grad_outputs(outputs_template)
        _check_array_types(
            grad_outputs, backend.CpuDevice(), 'generate_grad_outputs')
        return grad_outputs

    def _test_single_forward(self, inits, backend_config):
        inits = _get_initializers(inits)

        device = backend_config.device
        link = self._create_link(inits, device)

        inputs_np = self._generate_inputs()
        inputs_xp = tuple([device.send(i) for i in inputs_np])
        input_vars = tuple([chainer.Variable(i) for i in inputs_xp])

        actual_outputs_var = self._forward(link, input_vars, backend_config)

        cpu_device = backend.CpuDevice()
        actual_outputs_xp = [v.array for v in actual_outputs_var]
        actual_outputs_np = [cpu_device.send(arr) for arr in actual_outputs_xp]

        params = []
        for param_name in self.param_names:
            param = getattr(link, param_name, None)
            if param is None:
                raise RuntimeError(
                    'Link does not have a parameter named \'{}\'.'.format(
                        param_name))
            params.append(param)

        params_np = []
        for param in params:
            param_xp = param.array
            param_np = cpu_device.send(param_xp)
            params_np.append(param_np)

        expected_outputs_np = self._forward_expected(inputs_np, params_np)

        for e, a in zip(expected_outputs_np, actual_outputs_np):
            array_module.assert_allclose(e, a, **self.check_forward_options)

    def _test_single_backward(self, inits, backend_config):
        inits = _get_initializers(inits)

        _, _, outputs = self._generate_initialized_link(inits, backend_config)

        device = backend_config.device

        def f(*args):
            link = self._create_link(inits, device)
            return self._forward(link, args, backend_config)

        def do_check():
            inputs = self._generate_inputs()
            grad_outputs = self._generate_grad_outputs(outputs)

            inputs = backend_config.get_array(inputs)
            grad_outputs = backend_config.get_array(grad_outputs)
            # inputs = self._to_noncontiguous_as_needed(inputs)
            # grad_outputs = self._to_noncontiguous_as_needed(grad_outputs)

            gradient_check.check_backward(
                f, inputs, grad_outputs, dtype=inputs[0].dtype,
                detect_nondifferentiable=self.dodge_nondifferentiable,
                **self.check_backward_options)

        if self.dodge_nondifferentiable:
            while True:
                try:
                    # TODO(hvy): Resample.
                    do_check()
                except gradient_check.NondifferentiableError:
                    continue
                else:
                    break
        else:
            do_check()

    def _test_single_initializers(self, inits, backend_config):
        inits_orig = inits
        inits = _get_initializers(inits)

        device = backend_config.device
        link = self._create_link(inits, device)

        inputs_np = self._generate_inputs()
        inputs_xp = tuple([device.send(i) for i in inputs_np])
        input_vars = tuple([chainer.Variable(i) for i in inputs_xp])

        self._forward(link, input_vars, backend_config)

        # All parameters of link should have been initialized.
        params = []
        for param_name in self.param_names:
            param = getattr(link, param_name, None)
            if param is None:
                raise RuntimeError(
                    'Link does not have a parameter named \'{}\'.'.format(
                        param_name))
            params.append(param)

        cpu_device = backend.CpuDevice()
        params_xp = [v.array for v in params]
        params_np = [cpu_device.send(arr) for arr in params_xp]

        expected_inits, defaulted_indices = _get_expected_initializers(
            inits_orig, default_init=self.default_initializer,
            return_defaulted_indices=True)

        assert len(expected_inits) == len(inits)
        assert len(expected_inits) == len(params_np)

        for i in range(len(inits)):
            if i in defaulted_indices:
                continue

            param_np = params_np[i]
            expected_init = expected_inits[i]
            expected_np = numpy.empty_like(param_np)
            expected_init(expected_np)
            array_module.assert_allclose(expected_np, param_np)


def _initializer_iterator(params_inits, fill_init=None):
    if sys.version_info[0] < 3:
        zip_longest = itertools.izip_longest
    else:
        zip_longest = itertools.zip_longest
    return zip_longest(*params_inits, fillvalue=fill_init)


def _check_generated_initializer(init):
    if isinstance(init, InitializerPair):
        return
    initializers._check_is_initializer_like(init)


def _check_generated_initializers(params_inits):
    for param_inits in params_inits:
        for init in param_inits:
            _check_generated_initializer(init)


def _get_initializers(inits):
    assert isinstance(inits, tuple)

    ret = []
    for init in inits:
        if isinstance(init, InitializerPair):
            init = init.first
        ret.append(init)
    return tuple(ret)


def _get_expected_initializers(
        inits, default_init=None, return_defaulted_indices=False):
    assert isinstance(inits, tuple)
    assert default_init is None or isinstance(
        default_init, chainer.Initializer)

    ret = []
    if return_defaulted_indices:
        indices = []

    for i, init in enumerate(inits):
        if isinstance(init, InitializerPair):
            init = init.second
            if init is None:
                raise ValueError(
                    'Expected initializer in a InitializerPair should not be '
                    'None.')
        if init is None:
            init = default_init
            indices.append(i)
        elif not isinstance(init, chainer.Initializer):
            init = chainer.initializers._get_initializer(init)

        assert init is None or isinstance(init, chainer.Initializer)

        ret.append(init)

    ret = tuple(ret)
    if return_defaulted_indices:
        ret = ret, tuple(indices)
    return ret


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
