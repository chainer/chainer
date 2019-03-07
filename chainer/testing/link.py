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

    The initializer-like objects can be accessed via the ``argument_value`` and
    the ``expected_initializer`` attributes.

    When implementing ``LinkTestCase``, instances of this class can be included
    in lists listing all initializer-like objects that should be tested in the
    initializers test.
    In that case, the first element should correspond to the initializer-like
    argument passed to the constructor of the link, and the second element
    correspond to the actual initializer-like object used by the link.
    In many cases, they are the same, except when passing ``None`` to the
    constructor.
    """

    def __init__(self, argument_value, expected_initializer):
        if argument_value is not None:
            initializers._check_is_initializer_like(argument_value)
        if expected_initializer is not None:
            initializers._check_is_initializer_like(expected_initializer)

        self.argument_value = argument_value
        self.expected_initializer = expected_initializer


class LinkTestCase(unittest.TestCase):

    """A base class for link test cases.

    Link test cases can inherit from this class to define a set of link tests.

    .. rubric:: Required methods

    Each concrete class must at least override the following methods.

    ``generate_inputs(self)``
        Returns a tuple of input arrays of type :class:`numpy.ndarray`.

    ``create_link(self, initializers)``
        Returns a link. The link is typically initialized with the given
        initializer-likes ``initializers``. ``initializers`` is a tuple of
        same length as the number of parameters and contains initializer-likes
        returned by either ``generate_params`` or ``get_initializers``
        depending on the test being run.

    .. rubric:: Optional methods

    Additionally, the concrete class can override the following methods. Some
    must be overridden depending on the skip flags  ``skip_forward_test``,
    ``skip_backward_test`` and ``skip_initializers_test``.

    ``generate_params(self)``
        Returns a tuple of initializers-likes. The tuple should contain an
        initializer-like for each initializer-like argument, i.e. the
        parameters to the link constructor. These will be passed to
        ``create_link`` for the forward and backward tests.
        This method must be implemented if either ``skip_forward_test`` or
        ``skip_backward_test`` is ``False`` (forward or backward tests are
        executed).

    ``get_initializers(self)``
        Returns a tuple with the same length as the number of initializers that
        the constructor of the link accepts. Each element in the tuple is a
        container itself, listing all initializers-likes that should be tested.
        Each initializer-like in the tuple is tested one at a time by being
        passed to ``create_link``. When the length of the tuple is greater than
        one (i.e. if the link accepts multiple initializers), the ones not
        being tested are replaced by ``default_initializer``.
        Initializer-likes returned here should be deterministic since test will
        invoke them multiple times to test the correctness.
        For testing ``None`` as initializer-like arguments, one should wrap it
        in a ``InitializerPair`` as ``InitializerPair(None, expected)``, where
        the second argument is the expected initializer-like that the link is
        expected to use when passed ``None``. ``InitializerPair`` can be used
        to setup a test for any two initializer-likes where the first argument
        is passed to the link constructor and the second is the expected.
        Note that the expected cannot be ``None``.
        This method must be implemented if ``skip_initializers_test`` is
        ``False`` (the initializers test is executed).

    ``forward(self, link, inputs, device)``
        Implements the target forward function.
        ``link`` is a link created by ``create_link`` and
        ``inputs`` is a tuple of :class:`~chainer.Variable`\\ s.
        This method is expected to return the output
        :class:`~chainer.Variable`\\ s with the same array types as the inputs.
        ``device`` is the device corresponding to the input arrays.
        A default implementation is provided for links that only takes the
        inputs defined in ``generate_inputs`` (wrapped in
        :class:`~chainer.Variable`\\ s) and returns nothing but output
        :class:`~chainer.Variable`\\ s in its forward computation.

    ``forward_expected(self, inputs, params)``
        Implements the expectation of the target forward function.
        ``inputs`` and ``params`` are tuples of :class:`numpy.ndarray`\\ s.
        This method is expected to return the output
        :class:`numpy.ndarray`\\ s.
        This method must be implemented if either ``skip_forward_test`` or
        ``skip_backward_test`` is ``False`` (forward or backward tests are
        executed).

    ``before_test(self, test_name)``
        A callback method called before each test.
        Typically a skip logic is implemented by conditionally raising
        :class:`unittest.SkipTest`.
        ``test_name`` is one of ``'test_forward'``, ``'test_backward'``, and
        ``'test_initializers'``.

    ``generate_grad_outputs(self, outputs_template)``
        Returns a tuple of output gradient arrays of type
        :class:`numpy.ndarray`.
        ``outputs_template`` is a tuple of template arrays. The returned arrays
        are expected to have the same shapes and dtypes as the template arrays.

    .. rubric:: Attributes

    The concrete class can override the following attributes to control the
    behavior of the tests.

    ``skip_forward_test`` (bool):
        Whether to skip forward computation test. ``False`` by default.

    ``skip_backward_test`` (bool):
        Whether to skip backward computation test. ``False`` by default.

    ``skip_initializers_test`` (bool):
        Whether to skip link initialization test. ``False`` by default.

    ``param_names`` (list of str):
        A list of strings with all the names of the parameters that should be
        tested. E.g. ``['gamma', 'beta']`` for the batch normalization link.
        ``[]`` by default.

    ``default_initializer`` (initializer-like):
        Initializer-like that is used to pad initializer-like tuples when
        the link accepts multiple initializer-likes.
        This attribute is only used for the initializers tests.
        ``None`` by default.

    ``dodge_nondifferentiable`` (bool):
        Enable non-differentiable point detection in numerical gradient
        calculation. If the data returned by
        ``generate_params``, ``create_link`` and ``generate_inputs`` turns out
        to be a non-differentiable point, the test will repeatedly resample
        those until a differentiable point will be finally sampled. ``False``
        by default.

    ``contiguous`` (None or 'C'):
        Specifies the contiguousness of incoming arrays (i.e. inputs,
        parameters and gradients. If ``None``, the
        arrays will be non-contiguous as long as possible. If ``'C'``, the
        arrays will be C-contiguous. ``None`` by default.

    .. note::

        This class assumes :func:`chainer.testing.inject_backend_tests`
        is used together. See the example below.

    .. admonition:: Example

        .. testcode::

            @chainer.testing.inject_backend_tests(
              None,
              [
                  {},  # CPU
                  {'use_cuda': True},  # GPU
              ])
            class TestLinear(chainer.testing.LinkTestCase):

                param_names = ['W', 'b']

                def generate_params(self):
                    initialW = numpy.random.uniform(
                        -1, 1, (3, 2)).astype(numpy.float32)
                    initial_bias = numpy.random.uniform(
                        -1, 1, (3,)).astype(numpy.float32)
                    return initialW, initial_bias

                def get_initializers(self):
                    initialW = [initializers.Constant(1), 2]
                    initial_bias = [initializers.Constant(2), 3,
                        chainer.testing.link.InitializerPair(None, 0)]
                    return initialW, initial_bias

                def generate_inputs(self):
                    x = numpy.random.uniform(
                        -1, 1, (1, 2)).astype(numpy.float32)
                    return x,

                def create_link(self, initializers):
                    initialW, initial_bias = initializers
                    link = chainer.links.Linear(
                        2, 3, initialW=initialW, initial_bias=initial_bias)
                    return link

                def forward(self, link, inputs, device):
                    x, = inputs
                    return link(x),

                def forward_expected(self, inputs, params):
                    x, = inputs
                    W, b = params
                    expected = x.dot(W.T) + b
                    return expected,

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

    # The default initializer is used to pad the list of initializers during
    # the initializer tests when enumerating all of them to test only a
    # single initializer at a time.
    default_initializer = None

    def before_test(self, test_name):
        pass

    def generate_params(self):
        raise NotImplementedError('generate_params is not implemented.')

    def get_initializers(self):
        raise NotImplementedError('get_initializers is not implemented.')

    def create_link(self, initializers):
        raise NotImplementedError('create_link is not implemented.')

    def generate_inputs(self):
        raise NotImplementedError('generate_inputs is not implemented.')

    def forward(self, link, inputs, device):
        outputs = link(*inputs)
        if not isinstance(outputs, tuple):
            outputs = outputs,
        return outputs

    def forward_expected(self, inputs, params):
        raise NotImplementedError('forward_expected() is not implemented.')

    def generate_grad_outputs(self, outputs_template):
        grad_outputs = tuple([
            numpy.random.uniform(-1, 1, a.shape).astype(a.dtype)
            for a in outputs_template])
        return grad_outputs

    def test_forward(self, backend_config):
        """Tests forward computation."""

        if self.skip_forward_test:
            raise unittest.SkipTest('skip_forward_test is set')

        self.before_test('test_forward')

        inits = self._generate_params()
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

        self.before_test('test_backward')

        # avoid cyclic import
        from chainer import gradient_check

        def do_check():
            inits = self._generate_params()
            link = self._create_initialized_link(inits, backend_config)

            def f(inputs, ps):
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

        params_inits = self._get_initializers()

        default_init = self.default_initializer
        if default_init is not None:
            initializers._check_is_initializer_like(default_init)

        for i_param, param_inits in enumerate(params_inits):
            # When testing an initializer for a particular parameter, other
            # initializers are set to the default initializer.

            inits = [default_init, ] * len(params_inits)

            for init in param_inits:
                inits[i_param] = init
                self._test_single_initializer(i_param, inits, backend_config)

    def _test_single_initializer(self, i_param, inits, backend_config):
        # Given a set of initializer constructor arguments for the link, create
        # and initialize a link with those arguments. `i_param` holds the index
        # of the argument that should be tested among these.
        inits_orig = inits
        inits = [_get_initializer_argument_value(i) for i in inits]
        link = self._create_initialized_link(inits, backend_config)

        # Extract the parameters from the initialized link.
        params = _get_link_params(link, self.param_names)

        # Convert the parameter of interest into a NumPy ndarray.
        cpu_device = backend.CpuDevice()
        param = params[i_param]
        param_xp = param.array
        param_np = cpu_device.send(param_xp)

        # The expected values of the parameter is decided by the given
        # initializer. If the initializer is `None`, it should have been
        # wrapped in a InitializerPair along with the expected initializer that
        # the link should default to in case of `None`.
        #
        # Note that for this to work, the expected parameter must be inferred
        # deterministically.
        expected_init = _get_expected_initializer(inits_orig[i_param])
        expected_np = numpy.empty_like(param_np)
        expected_init(expected_np)

        # Compare the values of the expected and actual parameter.
        test._check_forward_output_arrays_equal(
            expected_np, param_np, 'forward', LinkTestError,
            **self.check_initializers_options)

    def _generate_params(self):
        params_init = self.generate_params()
        if not isinstance(params_init, (tuple, list)):
            raise TypeError(
                '`generate_params` must return a tuple or a list.')
        for init in params_init:
            _check_generated_initializer(init)
        return params_init

    def _get_initializers(self):
        params_inits = self.get_initializers()
        if not isinstance(params_inits, (tuple, list)):
            raise TypeError(
                '`get_initializers` must return a tuple or a list.')
        for param_inits in params_inits:
            if not isinstance(param_inits, (tuple, list)):
                raise TypeError(
                    '`get_initializers` must return a tuple or a list of '
                    'tuples or lists.')
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
        inits = [_get_initializer_argument_value(i) for i in inits]
        link = self._create_link(inits, backend_config)

        # Generate inputs and compute a forward pass to initialize the
        # parameters.
        inputs_np = self._generate_inputs()
        inputs_xp = backend_config.get_array(inputs_np)
        inputs_xp = self._to_noncontiguous_as_needed(inputs_xp)
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
        init = init.expected_initializer
    initializers._check_is_initializer_like(init)


def _get_initializer_argument_value(init):
    # Returns the initializer that should be passed to the link constructor.

    if isinstance(init, InitializerPair):
        return init.argument_value
    return init


def _get_expected_initializer(init):
    # Returns the expected initializer for the given initializer.

    if isinstance(init, InitializerPair):
        init = init.expected_initializer

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
