import unittest

import numpy

import chainer
from chainer import backend
from chainer import initializers
from chainer.testing import array as array_module
from chainer.testing import test


class LinkTestError(test.TestError):
    """Raised when the target link is implemented incorrectly."""
    pass


class InitializerPair(object):

    """Class to hold a pair of initializer-like objects.

    The initializer-like objects can be accessed via the ``first`` and the
    ``second`` attributes.

    When implementing ``LinkTestCase``, instances of this class can be included
    in lists listing all initializer-like objects that should be tested.
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
    """A base class for link test cases.

    Link test cases can inherit from this class to define a set of link
    tests.

    .. seealso:: :class:`~chainer.testing.FunctionTestCase`

    """

    backend_config = None
    check_forward_options = {}
    check_backward_options = {}
    check_initializers_options = {}
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
        """Return the default initializer.

        The default initializer is used to pad the list of initializers during
        the initializer tests when enumerating all of them to test only a
        single initializer at a time.
        """
        return self._default_initializer

    @default_initializer.setter
    def default_initializer(self, default_initializer):
        """Sets the default initializer."""
        initializers._check_is_initializer_like(default_initializer)
        self._default_initializer = default_initializer

    def before_test(self, test_name):
        """Is a method that is called before each test method.

        It is called before ``'test_forward'``, ``'test_backward'`` and
        ``'test_initializers'``.

        This method can be overridden for any pre-test setup such as
        tolerance configurations.
        """
        pass

    def generate_forward_backward_initializers(self):
        """Returns initializers for ``'test_forward'`` and ``'test_backward'``.

        Returns:
            A tuple of initializers. One for each initializer argument to the
            link constructor.
        """
        raise NotImplementedError(
            'generate_forward_backward_initializers is not implemented.')

    def generate_initializers(self):
        """Returns initializers for ``'test_initializers'``.

        Returns:
            A tuple of lists, each list containing all initializers to be
            tested for a particular parameter.
            The length of the tuple should be the same as the number of
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

    def forward(self, link, inputs, device):
        """Computes and returns the result of a forward pass."""
        outputs = link(*inputs)
        if not isinstance(outputs, tuple):
            outputs = outputs,
        return outputs

    def forward_expected(self, inputs, params):
        """Returns the expected results of a forward pass."""
        raise NotImplementedError('forward_expected() is not implemented.')

    def generate_grad_outputs(self, outputs_template):
        """Returns upstream gradients."""
        grad_outputs = tuple([
            numpy.random.uniform(-1, 1, a.shape).astype(a.dtype)
            for a in outputs_template])
        return grad_outputs

    def test_forward(self, backend_config):
        """Tests forward computation."""

        if self.skip_forward_test:
            raise unittest.SkipTest('skip_forward_test is set')

        self.before_test('test_forward')

        inits = self._generate_forward_backward_initializers()
        link = self._create_link(inits, backend_config)

        inputs_np = self._generate_inputs()
        inputs_xp = backend_config.get_array(inputs_np)
        inputs_xp = self._to_noncontiguous_as_needed(inputs_xp)
        input_vars = tuple([chainer.Variable(i) for i in inputs_xp])
        # Compute forward of the link and initialize its parameters.
        output_vars = self._forward(link, input_vars, backend_config)
        outputs_xp = [v.array for v in output_vars]

        # Expected outputs are computed on the CPU so inputs and parameters
        # must be transferred.
        cpu_device = backend.CpuDevice()
        params = _get_link_params(link, self.param_names)
        params_np = [cpu_device.send(p.array) for p in params]

        expected_outputs_np = self._forward_expected(inputs_np, params_np)

        test._check_forward_output_arrays_equal(
            expected_outputs_np, outputs_xp,
            'forward', LinkTestError, **self.check_forward_options)

    def test_backward(self, backend_config):
        """Tests backward computation."""

        if self.skip_backward_test:
            raise unittest.SkipTest('skip_backward_test is set')

        # avoid cyclic import
        from chainer import gradient_check

        def do_check():
            self.before_test('test_backward')

            inits = self._generate_forward_backward_initializers()

            def f(inputs, ps):
                link = self._create_initialized_link(inits, backend_config)
                with link.init_scope():
                    for param_name, p in zip(self.param_names, ps):
                        setattr(link, param_name, p)
                return self._forward(link, inputs, backend_config)

            link, inputs, outputs = self._create_initialized_link(
                inits, backend_config, return_inputs_outputs=True)

            params = _get_link_params(link, self.param_names)
            params = [p.array for p in params]

            cpu_device = backend.CpuDevice()
            outputs = [cpu_device.send(output) for output in outputs]
            grad_outputs = self._generate_grad_outputs(outputs)
            grad_outputs = backend_config.get_array(grad_outputs)

            inputs = self._to_noncontiguous_as_needed(inputs)
            params = self._to_noncontiguous_as_needed(params)
            grad_outputs = self._to_noncontiguous_as_needed(grad_outputs)

            with LinkTestError.raise_if_fail(
                    'backward is not implemented correctly'):
                gradient_check._check_backward_with_params(
                    f, inputs, grad_outputs, params=params,
                    dtype=numpy.float64,
                    detect_nondifferentiable=self.dodge_nondifferentiable,
                    **self.check_backward_options)

        if self.dodge_nondifferentiable:
            while True:
                try:
                    do_check()
                except gradient_check.NondifferentiableError:
                    continue
                else:
                    break
        else:
            do_check()

    def test_initializers(self, backend_config):
        """Tests that the parameters of a links are correctly initialized."""
        if self.skip_initializers_test:
            raise unittest.SkipTest('skip_initializers_test is set')

        self.before_test('test_initializers')

        params_inits = self._generate_initializers()

        for i_param, param_inits in enumerate(params_inits):
            # When testing an initializer for a particular parameter, other
            # initializers are set to the default initializer.
            inits = [self.default_initializer, ] * len(params_inits)

            for init in param_inits:
                inits[i_param] = init
                self._test_single_initializer(i_param, inits, backend_config)

    def _test_single_initializer(self, i_param, inits, backend_config):
        inits_orig = inits
        inits = [_get_initializer(i) for i in inits]
        link = self._create_initialized_link(inits, backend_config)

        params = _get_link_params(link, self.param_names)

        cpu_device = backend.CpuDevice()
        param = params[i_param]
        param_xp = param.array
        param_np = cpu_device.send(param_xp)

        expected_init = _get_expected_initializer(inits_orig[i_param])
        expected_np = numpy.empty_like(param_np)
        expected_init(expected_np)

        test._check_forward_output_arrays_equal(
            expected_np, param_np, 'forward', LinkTestError,
            **self.check_initializers_options)

    def _generate_forward_backward_initializers(self):
        params_init = self.generate_forward_backward_initializers()
        for init in params_init:
            _check_generated_initializer(init)
        return params_init

    def _generate_initializers(self):
        params_inits = self.generate_initializers()
        for param_inits in params_inits:
            for init in param_inits:
                _check_generated_initializer(init)
        return params_inits

    def _create_link(self, initializers, backend_config):
        link = self.create_link(initializers)
        if not isinstance(link, chainer.Link):
            raise TypeError(
                '`create_link` must return a chainer.Link object.')

        link.to_device(backend_config.device)

        return link

    def _create_initialized_link(
            self, inits, backend_config, return_inputs_outputs=False):
        inits = [_get_initializer(i) for i in inits]
        link = self._create_link(inits, backend_config)

        # Generate inputs and compute a forward pass to initialize the
        # parameters.
        # TODO(hvy): Allow testing initializers without implementing
        # generate_forward_backward_initializers.
        inputs_np = self._generate_inputs()
        inputs_xp = backend_config.get_array(inputs_np)
        input_vars = [chainer.Variable(i) for i in inputs_xp]
        output_vars = self._forward(link, input_vars, backend_config)

        link.cleargrads()

        ret = link

        if return_inputs_outputs:
            outputs_xp = [v.array for v in output_vars]
            ret = ret, inputs_xp, outputs_xp

        return ret

    def _generate_inputs(self):
        inputs = self.generate_inputs()

        test._check_array_types(inputs, backend.CpuDevice(), 'generate_inputs')

        return inputs

    def _forward(self, link, inputs, backend_config):
        assert all(isinstance(x, chainer.Variable) for x in inputs)

        with backend_config:
            outputs = self.forward(link, inputs, backend_config.device)
        test._check_variable_types(
            outputs, backend_config.device, 'forward', LinkTestError)

        return outputs

    def _forward_expected(self, inputs, params):
        assert all(isinstance(x, numpy.ndarray) for x in inputs)
        assert all(isinstance(x, numpy.ndarray) for x in params)

        outputs = self.forward_expected(inputs, params)
        test._check_array_types(inputs, backend.CpuDevice(), 'test_forward')

        return outputs

    def _generate_grad_outputs(self, outputs_template):
        assert all(isinstance(x, numpy.ndarray) for x in outputs_template)

        grad_outputs = self.generate_grad_outputs(outputs_template)
        test._check_array_types(
            grad_outputs, backend.CpuDevice(), 'generate_grad_outputs')

        return grad_outputs

    def _to_noncontiguous_as_needed(self, contig_arrays):
        if self.contiguous is None:
            # non-contiguous
            return array_module._as_noncontiguous_array(contig_arrays)
        if self.contiguous == 'C':
            # C-contiguous
            return contig_arrays
        assert False, (
            'Invalid value of `contiguous`: {}'.format(self.contiguous))


def _check_generated_initializer(init):
    if isinstance(init, InitializerPair):
        init = init.second
        if init is None:
            raise TypeError(
                'Expected (second) initializer in a InitializerPair should '
                'not be None.')
        return
    initializers._check_is_initializer_like(init)


def _get_initializer(init):
    # Returns the initializer that should be passed to the link constructor.

    if isinstance(init, InitializerPair):
        return init.first
    return init


def _get_expected_initializer(init):
    # Returns the expected initializer for the given initializer.

    if isinstance(init, InitializerPair):
        init = init.second

    assert init is not None

    if not isinstance(init, chainer.Initializer):
        init = chainer.initializers._get_initializer(init)
    return init


def _get_link_params(link, param_names):
    params = []
    for name in param_names:
        param = getattr(link, name, None)
        if param is None:
            raise LinkTestError.fail(
                'Link does not have a parameter named \'{}\'.'.format(name))
        params.append(param)
    return params
