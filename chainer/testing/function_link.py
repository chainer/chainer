import contextlib
import typing as tp  # NOQA
import unittest

import numpy
import six

import chainer
from chainer import backend
from chainer import initializers
from chainer.testing import array as array_module
from chainer import utils


class _TestError(AssertionError):

    """Parent class to Chainer test errors."""

    @classmethod
    def check(cls, expr, message):
        if not expr:
            raise cls(message)

    @classmethod
    def fail(cls, message, exc=None):
        if exc is not None:
            utils._raise_from(cls, message, exc)
        raise cls(message)

    @classmethod
    @contextlib.contextmanager
    def raise_if_fail(cls, message, error_types=AssertionError):
        try:
            yield
        except error_types as e:
            cls.fail(message, e)


class FunctionTestError(_TestError):
    """Raised when the target function is implemented incorrectly."""
    pass


class LinkTestError(_TestError):
    """Raised when the target link is implemented incorrectly."""
    pass


class InitializerArgument(object):

    """Class to hold a pair of initializer argument value and actual
    initializer-like.

    This class is meant to be included in the return value from
    :meth:`chainer.testing.LinkTestCase.get_initializers` in
    :class:`chainer.testing.LinkTestCase` if the argument and the actual
    initializer in the link do not directly correspond.
    In that case, the first element should correspond to the argument passed to
    the constructor of the link, and the second element correspond to the
    actual initializer-like object used by the link.
    """

    def __init__(self, argument_value, expected_initializer):
        if expected_initializer is None:
            raise ValueError('Expected initialized cannot be None.')
        initializers._check_is_initializer_like(expected_initializer)

        self.argument_value = argument_value
        self.expected_initializer = expected_initializer


class FunctionTestBase(object):

    backend_config = None
    check_forward_options = None
    check_backward_options = None
    check_double_backward_options = None
    skip_forward_test = False
    skip_backward_test = False
    skip_double_backward_test = False
    dodge_nondifferentiable = False
    numerical_grad_dtype = numpy.float64
    contiguous = None

    def __init__(self, *args, **kwargs):
        super(FunctionTestBase, self).__init__(*args, **kwargs)
        self.check_forward_options = {}
        self.check_backward_options = {}
        self.check_double_backward_options = {}

    def before_test(self, test_name):
        pass

    def forward(self, inputs, device):
        raise NotImplementedError('forward() is not implemented.')

    def forward_expected(self, inputs):
        raise NotImplementedError('forward_expected() is not implemented.')

    def generate_inputs(self):
        raise NotImplementedError('generate_inputs() is not implemented.')

    def generate_grad_outputs(self, outputs_template):
        grad_outputs = tuple([
            numpy.random.uniform(-1, 1, a.shape).astype(a.dtype)
            for a in outputs_template])
        return grad_outputs

    def generate_grad_grad_inputs(self, inputs_template):
        grad_grad_inputs = tuple([
            numpy.random.uniform(-1, 1, a.shape).astype(a.dtype)
            for a in inputs_template])
        return grad_grad_inputs

    def check_forward_outputs(self, outputs, expected_outputs):
        assert isinstance(outputs, tuple)
        assert isinstance(expected_outputs, tuple)
        assert all(isinstance(a, chainer.get_array_types()) for a in outputs)
        assert all(
            isinstance(a, chainer.get_array_types()) for a in expected_outputs)
        _check_arrays_equal(
            outputs, expected_outputs, FunctionTestError,
            **self.check_forward_options)

    def _to_noncontiguous_as_needed(self, contig_arrays):
        if self.contiguous is None:
            # non-contiguous
            return array_module._as_noncontiguous_array(contig_arrays)
        if self.contiguous == 'C':
            # C-contiguous
            return contig_arrays
        assert False, (
            'Invalid value of `contiguous`: {}'.format(self.contiguous))

    def _generate_inputs(self):
        inputs = self.generate_inputs()
        _check_array_types(inputs, backend.CpuDevice(), 'generate_inputs')
        return inputs

    def _generate_grad_outputs(self, outputs_template):
        grad_outputs = self.generate_grad_outputs(outputs_template)
        _check_array_types(
            grad_outputs, backend.CpuDevice(), 'generate_grad_outputs')
        return grad_outputs

    def _generate_grad_grad_inputs(self, inputs_template):
        grad_grad_inputs = self.generate_grad_grad_inputs(inputs_template)
        _check_array_types(
            grad_grad_inputs, backend.CpuDevice(), 'generate_grad_grad_inputs')
        return grad_grad_inputs

    def _forward_expected(self, inputs):
        outputs = self.forward_expected(inputs)
        _check_array_types(
            outputs, backend.CpuDevice(), 'forward_expected')
        return outputs

    def _forward(self, inputs, backend_config):
        assert all(isinstance(a, chainer.Variable) for a in inputs)
        with backend_config:
            outputs = self.forward(inputs, backend_config.device)
        _check_variable_types(
            outputs, backend_config.device, 'forward', FunctionTestError)
        return outputs

    def run_test_forward(self, backend_config):
        # Runs the forward test.

        if self.skip_forward_test:
            raise unittest.SkipTest('skip_forward_test is set')

        self.backend_config = backend_config
        self.test_name = 'test_forward'
        self.before_test(self.test_name)

        cpu_inputs = self._generate_inputs()
        cpu_inputs = self._to_noncontiguous_as_needed(cpu_inputs)
        inputs_copied = [a.copy() for a in cpu_inputs]

        # Compute expected outputs
        cpu_expected = self._forward_expected(cpu_inputs)

        # Compute actual outputs
        inputs = backend_config.get_array(cpu_inputs)
        inputs = self._to_noncontiguous_as_needed(inputs)
        outputs = self._forward(
            tuple([
                chainer.Variable(a, requires_grad=a.dtype.kind == 'f')
                for a in inputs]),
            backend_config)

        # Check inputs has not changed
        indices = []
        for i in range(len(inputs)):
            try:
                array_module.assert_allclose(
                    inputs_copied[i], inputs[i], atol=0, rtol=0)
            except AssertionError:
                indices.append(i)

        if indices:
            f = six.StringIO()
            f.write(
                'Input arrays have been modified during forward.\n'
                'Indices of modified inputs: {}\n'
                'Input array shapes and dtypes: {}\n'.format(
                    ', '.join(str(i) for i in indices),
                    utils._format_array_props(inputs)))
            for i in indices:
                f.write('\n')
                f.write('Input[{}]:\n'.format(i))
                f.write('Original:\n')
                f.write(str(inputs_copied[i]))
                f.write('\n')
                f.write('After forward:\n')
                f.write(str(inputs[i]))
                f.write('\n')
            FunctionTestError.fail(f.getvalue())

        self.check_forward_outputs(
            tuple([var.array for var in outputs]),
            cpu_expected)

    def run_test_backward(self, backend_config):
        # Runs the backward test.

        if self.skip_backward_test:
            raise unittest.SkipTest('skip_backward_test is set')

        # avoid cyclic import
        from chainer import gradient_check

        self.backend_config = backend_config
        self.test_name = 'test_backward'
        self.before_test(self.test_name)

        def f(*args):
            return self._forward(args, backend_config)

        def do_check():
            inputs = self._generate_inputs()
            outputs = self._forward_expected(inputs)
            grad_outputs = self._generate_grad_outputs(outputs)

            inputs = backend_config.get_array(inputs)
            grad_outputs = backend_config.get_array(grad_outputs)
            inputs = self._to_noncontiguous_as_needed(inputs)
            grad_outputs = self._to_noncontiguous_as_needed(grad_outputs)

            with FunctionTestError.raise_if_fail(
                    'backward is not implemented correctly'):
                gradient_check.check_backward(
                    f, inputs, grad_outputs, dtype=self.numerical_grad_dtype,
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

    def run_test_double_backward(self, backend_config):
        # Runs the double-backward test.

        if self.skip_double_backward_test:
            raise unittest.SkipTest('skip_double_backward_test is set')

        # avoid cyclic import
        from chainer import gradient_check

        self.backend_config = backend_config
        self.test_name = 'test_double_backward'
        self.before_test(self.test_name)

        def f(*args):
            return self._forward(args, backend_config)

        def do_check():
            inputs = self._generate_inputs()
            outputs = self._forward_expected(inputs)
            grad_outputs = self._generate_grad_outputs(outputs)
            grad_grad_inputs = self._generate_grad_grad_inputs(inputs)

            # Drop ggx corresponding to non-differentiable inputs.
            # Generated `grad_grad_inputs`, the upstream gradients for the
            # double backward test, may contain `None` for omitted gradients.
            # These must be propagated to the gradient check.
            grad_grad_inputs = [
                ggx for ggx in grad_grad_inputs
                if (ggx is None or ggx.dtype.kind == 'f')]

            inputs = backend_config.get_array(inputs)
            grad_outputs = backend_config.get_array(grad_outputs)
            grad_grad_inputs = backend_config.get_array(grad_grad_inputs)
            inputs = self._to_noncontiguous_as_needed(inputs)
            grad_outputs = self._to_noncontiguous_as_needed(grad_outputs)
            grad_grad_inputs = (
                self._to_noncontiguous_as_needed(grad_grad_inputs))

            with backend_config:
                with FunctionTestError.raise_if_fail(
                        'double backward is not implemented correctly'):
                    gradient_check.check_double_backward(
                        f, inputs, grad_outputs, grad_grad_inputs,
                        dtype=self.numerical_grad_dtype,
                        detect_nondifferentiable=self.dodge_nondifferentiable,
                        **self.check_double_backward_options)

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


class FunctionTestCase(FunctionTestBase, unittest.TestCase):
    """A base class for function test cases.

    Function test cases can inherit from this class to define a set of function
    tests.

    .. rubric:: Required methods

    Each concrete class must at least override the following three methods.

    ``forward(self, inputs, device)``
        Implements the target forward function.
        ``inputs`` is a tuple of :class:`~chainer.Variable`\\ s.
        This method is expected to return the output
        :class:`~chainer.Variable`\\ s with the same array types as the inputs.
        ``device`` is the device corresponding to the input arrays.

    ``forward_expected(self, inputs)``
        Implements the expectation of the target forward function.
        ``inputs`` is a tuple of :class:`numpy.ndarray`\\ s.
        This method is expected to return the output
        :class:`numpy.ndarray`\\ s.

    ``generate_inputs(self)``
        Returns a tuple of input arrays of type :class:`numpy.ndarray`.

    .. rubric:: Optional methods

    Additionally the concrete class can override the following methods.

    ``before_test(self, test_name)``
        A callback method called before each test.
        Typically a skip logic is implemented by conditionally raising
        :class:`unittest.SkipTest`.
        ``test_name`` is one of ``'test_forward'``, ``'test_backward'``, and
        ``'test_double_backward'``.

    ``generate_grad_outputs(self, outputs_template)``
        Returns a tuple of output gradient arrays of type
        :class:`numpy.ndarray` or ``None`` for omitted the gradients.
        ``outputs_template`` is a tuple of template arrays. The returned arrays
        are expected to have the same shapes and dtypes as the template arrays.

    ``generate_grad_grad_inputs(self, inputs_template)``
        Returns a tuple of the second order input gradient arrays of type
        :class:`numpy.ndarray` or ``None`` for omitted gradients.
        ``input_template`` is a tuple of template arrays. The returned arrays
        are expected to have the same shapes and dtypes as the template arrays.

    ``check_forward_outputs(self, outputs, expected_outputs)``
        Implements check logic of forward outputs. Typically additional check
        can be done after calling ``super().check_forward_outputs``.
        ``outputs`` and ``expected_outputs`` are tuples of arrays.
        In case the check fails, ``FunctionTestError`` should be raised.

    .. rubric:: Configurable attributes

    The concrete class can override the following attributes to control the
    behavior of the tests.

    ``skip_forward_test`` (bool):
        Whether to skip forward computation test. ``False`` by default.

    ``skip_backward_test`` (bool):
        Whether to skip backward computation test. ``False`` by default.

    ``skip_double_backward_test`` (bool):
        Whether to skip double-backward computation test. ``False`` by default.

    ``dodge_nondifferentiable`` (bool):
        Enable non-differentiable point detection in numerical gradient
        calculation. If the inputs returned by ``generate_inputs`` turns
        out to be a non-differentiable point, the test will repeatedly resample
        inputs until a differentiable point will be finally sampled.
        ``False`` by default.

    ``numerical_grad_dtype`` (dtype):
        Input arrays are casted to this dtype when calculating the numerical
        gradients. It is ``float64`` by default, no matter what the original
        input dtypes were, to maximize precision.

    ``contiguous`` (None or 'C'):
        Specifies the contiguousness of incoming arrays (i.e. inputs, output
        gradients, and the second order input gradients). If ``None``, the
        arrays will be non-contiguous as long as possible. If ``'C'``, the
        arrays will be C-contiguous. ``None`` by default.

    .. rubric:: Passive attributes

    These attributes are automatically set.

    ``test_name`` (str):
        The name of the test being run. It is one of ``'test_forward'``,
        ``'test_backward'``, and ``'test_double_backward'``.

    ``backend_config`` (:class:`~chainer.testing.BackendConfig`):
        The backend configuration.

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
          class TestReLU(chainer.testing.FunctionTestCase):

              # ReLU function has a non-differentiable point around zero, so
              # dodge_nondifferentiable should be set to True.
              dodge_nondifferentiable = True

              def generate_inputs(self):
                  x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
                  return x,

              def forward(self, inputs, device):
                  x, = inputs
                  return F.relu(x),

              def forward_expected(self, inputs):
                  x, = inputs
                  expected = x.copy()
                  expected[expected < 0] = 0
                  return expected,

    .. seealso:: :class:`~chainer.testing.LinkTestCase`

    """

    def test_forward(self, backend_config):
        """Tests forward computation."""
        self.run_test_forward(backend_config)

    def test_backward(self, backend_config):
        """Tests backward computation."""
        self.run_test_backward(backend_config)

    def test_double_backward(self, backend_config):
        """Tests double-backward computation."""
        self.run_test_double_backward(backend_config)


class _LinkTestBase(object):

    backend_config = None
    contiguous = None

    # List of parameter names represented as strings.
    # I.e. ('gamma', 'beta') for BatchNormalization.
    param_names = ()

    def before_test(self, test_name):
        pass

    def generate_params(self):
        raise NotImplementedError('generate_params is not implemented.')

    def generate_inputs(self):
        raise NotImplementedError('generate_inputs is not implemented.')

    def create_link(self, initializers):
        raise NotImplementedError('create_link is not implemented.')

    def forward(self, link, inputs, device):
        outputs = link(*inputs)
        if not isinstance(outputs, tuple):
            outputs = outputs,
        return outputs

    def check_forward_outputs(self, outputs, expected_outputs):
        assert isinstance(outputs, tuple)
        assert isinstance(expected_outputs, tuple)
        assert all(isinstance(a, chainer.get_array_types()) for a in outputs)
        assert all(
            isinstance(a, chainer.get_array_types()) for a in expected_outputs)
        _check_arrays_equal(
            outputs, expected_outputs, LinkTestError,
            **self.check_forward_options)

    def _generate_params(self):
        params_init = self.generate_params()
        if not isinstance(params_init, (tuple, list)):
            raise TypeError(
                '`generate_params` must return a tuple or a list.')
        for init in params_init:
            _check_generated_initializer(init)
        return params_init

    def _generate_inputs(self):
        inputs = self.generate_inputs()

        _check_array_types(inputs, backend.CpuDevice(), 'generate_inputs')

        return inputs

    def _create_link(self, initializers, backend_config):
        link = self.create_link(initializers)
        if not isinstance(link, chainer.Link):
            raise TypeError(
                '`create_link` must return a chainer.Link object.')

        link.to_device(backend_config.device)

        return link

    def _create_initialized_link(self, inits, backend_config):
        inits = [_get_initializer_argument_value(i) for i in inits]
        link = self._create_link(inits, backend_config)

        # Generate inputs and compute a forward pass to initialize the
        # parameters.
        inputs_np = self._generate_inputs()
        inputs_xp = backend_config.get_array(inputs_np)
        inputs_xp = self._to_noncontiguous_as_needed(inputs_xp)
        input_vars = [chainer.Variable(i) for i in inputs_xp]
        output_vars = self._forward(link, input_vars, backend_config)
        outputs_xp = [v.array for v in output_vars]

        link.cleargrads()

        return link, inputs_xp, outputs_xp

    def _forward(self, link, inputs, backend_config):
        assert all(isinstance(x, chainer.Variable) for x in inputs)

        with backend_config:
            outputs = self.forward(link, inputs, backend_config.device)
        _check_variable_types(
            outputs, backend_config.device, 'forward', LinkTestError)

        return outputs

    def _to_noncontiguous_as_needed(self, contig_arrays):
        if self.contiguous is None:
            # non-contiguous
            return array_module._as_noncontiguous_array(contig_arrays)
        if self.contiguous == 'C':
            # C-contiguous
            return contig_arrays
        assert False, (
            'Invalid value of `contiguous`: {}'.format(self.contiguous))


class LinkTestCase(_LinkTestBase, unittest.TestCase):

    """A base class for link forward and backward test cases.

    Link test cases can inherit from this class to define a set of link tests
    for forward and backward computations.

    .. rubric:: Required methods

    Each concrete class must at least override the following methods.

    ``generate_params(self)``
        Returns a tuple of initializers-likes. The tuple should contain an
        initializer-like for each initializer-like argument, i.e. the
        parameters to the link constructor. These will be passed to
        ``create_link``.

    ``create_link(self, initializers)``
        Returns a link. The link should be initialized with the given
        initializer-likes ``initializers``. ``initializers`` is a tuple of
        same length as the number of parameters.

    ``generate_inputs(self)``
        Returns a tuple of input arrays of type :class:`numpy.ndarray`.

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

    .. rubric:: Optional methods

    Each concrete class may override the following methods depending on the
    skip flags  ``skip_forward_test`` and  ``skip_backward_test``.

    ``before_test(self, test_name)``
        A callback method called before each test.
        Typically a skip logic is implemented by conditionally raising
        :class:`unittest.SkipTest`.
        ``test_name`` is one of ``'test_forward'`` and ``'test_backward'``.

    ``forward_expected(self, link, inputs)``
        Implements the expectation of the target forward function.
        ``link`` is the initialized link that was used to compute the actual
        forward which the results of this method will be compared against.
        The link is guaranteed to reside on the CPU.
        ``inputs`` is a tuple of :class:`numpy.ndarray`\\ s.
        This method is expected to return the output
        :class:`numpy.ndarray`\\ s.
        This method must be implemented if either ``skip_forward_test`` or
        ``skip_backward_test`` is ``False`` in which case forward or backward
        tests are executed.

    ``generate_grad_outputs(self, outputs_template)``
        Returns a tuple of output gradient arrays of type
        :class:`numpy.ndarray`.
        ``outputs_template`` is a tuple of template arrays. The returned arrays
        are expected to have the same shapes and dtypes as the template arrays.

    ``check_forward_outputs(self, outputs, expected_outputs)``
        Implements check logic of forward outputs. Typically additional check
        can be done after calling ``super().check_forward_outputs``.
        ``outputs`` and ``expected_outputs`` are tuples of arrays.
        In case the check fails, ``LinkTestError`` should be raised.

    .. rubric:: Attributes

    The concrete class can override the following attributes to control the
    behavior of the tests.

    ``param_names`` (tuple of str):
        A tuple of strings with all the names of the parameters that should be
        tested. E.g. ``('gamma', 'beta')`` for the batch normalization link.
        ``()`` by default.

    ``skip_forward_test`` (bool):
        Whether to skip forward computation test. ``False`` by default.

    ``skip_backward_test`` (bool):
        Whether to skip backward computation test. ``False`` by default.

    ``dodge_nondifferentiable`` (bool):
        Enable non-differentiable point detection in numerical gradient
        calculation. If the data returned by
        ``generate_params``, ``create_link`` and ``generate_inputs`` turns out
        to be a non-differentiable point, the test will repeatedly resample
        those until a differentiable point will be finally sampled. ``False``
        by default.

    ``numerical_grad_dtype`` (dtype):
        Input arrays are casted to this dtype when calculating the numerical
        gradients. It is ``float64`` by default, no matter what the original
        input dtypes were, to maximize precision.

    ``contiguous`` (None or 'C'):
        Specifies the contiguousness of incoming arrays (i.e. inputs,
        parameters and gradients. If ``None``, the
        arrays will be non-contiguous as long as possible. If ``'C'``, the
        arrays will be C-contiguous. ``None`` by default.

    .. note::

        This class assumes :func:`chainer.testing.inject_backend_tests`
        is used together. See the example below.

    .. note::

        When implementing :class:`~chainer.testing.LinkTestCase` and
        :class:`~chainer.testing.LinkInitializersTestCase` to test both
        forward/backward and initializers, it is often convenient to refactor
        out common logic in a separate class.

    .. admonition:: Example

        .. testcode::

            @chainer.testing.inject_backend_tests(
              None,
              [
                  {},  # CPU
                  {'use_cuda': True},  # GPU
              ])
            class TestLinear(chainer.testing.LinkTestCase):

                param_names = ('W', 'b')

                def generate_params(self):
                    initialW = numpy.random.uniform(
                        -1, 1, (3, 2)).astype(numpy.float32)
                    initial_bias = numpy.random.uniform(
                        -1, 1, (3,)).astype(numpy.float32)
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

                def forward_expected(self, link, inputs):
                    W = link.W.array
                    b = link.b.array
                    x, = inputs
                    expected = x.dot(W.T) + b
                    return expected,

    .. seealso::
        :class:`~chainer.testing.LinkInitializersTestCase`
        :class:`~chainer.testing.FunctionTestCase`

    """

    check_forward_options = None
    check_backward_options = None
    skip_forward_test = False
    skip_backward_test = False
    dodge_nondifferentiable = False
    numerical_grad_dtype = numpy.float64

    def __init__(self, *args, **kwargs):
        self.check_forward_options = {}
        self.check_backward_options = {}

        super(LinkTestCase, self).__init__(*args, **kwargs)

    def forward_expected(self, link, inputs):
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

        self.backend_config = backend_config
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

        # Expected outputs are computed on the CPU so the link must be
        # transferred.
        link.to_device(backend.CpuDevice())

        expected_outputs_np = self._forward_expected(link, inputs_np)

        self.check_forward_outputs(
            tuple(outputs_xp), expected_outputs_np)

    def test_backward(self, backend_config):
        """Tests backward computation."""

        if self.skip_backward_test:
            raise unittest.SkipTest('skip_backward_test is set')

        self.backend_config = backend_config
        self.before_test('test_backward')

        # avoid cyclic import
        from chainer import gradient_check

        def do_check():
            # Generate an initialized temporary link that is already forward
            # propagated. This link is only used to generate necessary data,
            # i.e. inputs, outputs and parameters for the later gradient check
            # and the link itself will be discarded.
            inits = self._generate_params()
            link, inputs, outputs = self._create_initialized_link(
                inits, backend_config)

            # Extract the parameter ndarrays from the initialized link.
            params = _get_link_params(link, self.param_names)
            params = [p.array for p in params]

            # Prepare inputs, outputs and upstream gradients for the gradient
            # check.
            cpu_device = backend.CpuDevice()
            outputs = [cpu_device.send(output) for output in outputs]
            grad_outputs = self._generate_grad_outputs(outputs)
            grad_outputs = backend_config.get_array(grad_outputs)

            inputs = self._to_noncontiguous_as_needed(inputs)
            params = self._to_noncontiguous_as_needed(params)
            grad_outputs = self._to_noncontiguous_as_needed(grad_outputs)

            # Create the link used for the actual forward propagation in the
            # gradient check.
            forward_link, _, _ = self._create_initialized_link(
                inits, backend_config)

            def forward(inputs, ps):

                # Use generated parameters.
                with forward_link.init_scope():
                    for param_name, p in zip(self.param_names, ps):
                        setattr(forward_link, param_name, p)

                return self._forward(forward_link, inputs, backend_config)

            with LinkTestError.raise_if_fail(
                    'backward is not implemented correctly'):
                gradient_check._check_backward_with_params(
                    forward, inputs, grad_outputs, params=params,
                    dtype=self.numerical_grad_dtype,
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

    def _forward_expected(self, link, inputs):
        assert all(isinstance(x, numpy.ndarray) for x in inputs)

        outputs = self.forward_expected(link, inputs)
        _check_array_types(inputs, backend.CpuDevice(), 'test_forward')

        return outputs

    def _generate_grad_outputs(self, outputs_template):
        assert all(isinstance(x, numpy.ndarray) for x in outputs_template)

        grad_outputs = self.generate_grad_outputs(outputs_template)
        _check_array_types(
            grad_outputs, backend.CpuDevice(), 'generate_grad_outputs')

        return grad_outputs


class LinkInitializersTestCase(_LinkTestBase, unittest.TestCase):

    """A base class for link parameter initializer test cases.

    Link test cases can inherit from this class to define a set of link tests
    for parameter initialization.

    .. rubric:: Required methods

    Each concrete class must at least override the following methods.

    ``generate_params(self)``
        Returns a tuple of initializers-likes. The tuple should contain an
        initializer-like for each initializer-like argument, i.e. the
        parameters to the link constructor. These will be passed to
        ``create_link``.

    ``create_link(self, initializers)``
        Returns a link. The link should be initialized with the given
        initializer-likes ``initializers``. ``initializers`` is a tuple of
        same length as the number of parameters.

    ``generate_inputs(self)``
        Returns a tuple of input arrays of type :class:`numpy.ndarray`.

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

    ``get_initializers(self)``
        Returns a tuple with the same length as the number of initializers that
        the constructor of the link accepts. Each element in the tuple is a
        container itself, listing all initializers-likes that should be tested.
        Each initializer-like in the tuple is tested one at a time by being
        passed to ``create_link``. When the length of the tuple is greater than
        one (i.e. if the link accepts multiple initializers), the ones not
        being tested are replaced by the ones returned by `generate_params`.
        Initializer-likes returned here should be deterministic since test will
        invoke them multiple times to test the correctness.

        For testing initializer arguments that can be non-initializer values
        such as ``None``, one can use the ``InitializerArgument``, defining a
        pair of the link constructor argument and actual initializer-like used
        by the link.
        This method must be implemented if ``skip_initializers_test`` is
        ``False`` in which case the initializers test is executed.

    .. rubric:: Optional methods

    Each concrete class may override the following methods.

    ``before_test(self, test_name)``
        A callback method called before each test.
        Typically a skip logic is implemented by conditionally raising
        :class:`unittest.SkipTest`.
        ``test_name`` is always of ``'test_initializers'``.

    .. rubric:: Attributes

    The concrete class can override the following attributes to control the
    behavior of the tests.

    ``param_names`` (list of str):
        A list of strings with all the names of the parameters that should be
        tested. E.g. ``['gamma', 'beta']`` for the batch normalization link.
        ``[]`` by default.

    ``contiguous`` (None or 'C'):
        Specifies the contiguousness of incoming arrays (i.e. inputs,
        parameters and gradients. If ``None``, the
        arrays will be non-contiguous as long as possible. If ``'C'``, the
        arrays will be C-contiguous. ``None`` by default.

    .. note::

        This class assumes :func:`chainer.testing.inject_backend_tests`
        is used together. See the example below.

    .. note::

        When implementing :class:`~chainer.testing.LinkTestCase` and
        :class:`~chainer.testing.LinkInitializersTestCase` to test both
        forward/backward and initializers, it is often convenient to refactor
        out common logic in a separate class.

    .. admonition:: Example

        .. testcode::

            @chainer.testing.inject_backend_tests(
              None,
              [
                  {},  # CPU
                  {'use_cuda': True},  # GPU
              ])
            class TestLinear(chainer.testing.LinkInitializersTestCase):

                param_names = ['W', 'b']

                def generate_params(self):
                    initialW = numpy.random.uniform(
                        -1, 1, (3, 2)).astype(numpy.float32)
                    initial_bias = numpy.random.uniform(
                        -1, 1, (3,)).astype(numpy.float32)
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

                def get_initializers(self):
                    initialW = [initializers.Constant(1), 2]
                    initial_bias = [initializers.Constant(2), 3,
                        chainer.testing.link.InitializerArgument(None, 0)]
                    return initialW, initial_bias

    .. seealso::
        :class:`~chainer.testing.LinkTestCase`
        :class:`~chainer.testing.FunctionTestCase`

    """

    check_initializers_options = None

    def __init__(self, *args, **kwargs):
        self.check_initializers_options = {}

        super(LinkInitializersTestCase, self).__init__(*args, **kwargs)

    def get_initializers(self):
        raise NotImplementedError('get_initializers is not implemented.')

    def test_initializers(self, backend_config):
        """Tests that the parameters of a links are correctly initialized."""

        self.backend_config = backend_config
        self.before_test('test_initializers')

        params_inits = self._get_initializers()

        # TODO(hvy): Reduce the number of loop iterations by checking
        # multiple parameters simultaneously.
        for i_param, param_inits in enumerate(params_inits):
            # When testing an initializer for a particular parameter, other
            # initializers are picked from generate_params.
            inits = self._generate_params()
            inits = list(inits)

            for init in param_inits:
                inits[i_param] = init
                self._test_single_initializer(i_param, inits, backend_config)

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

    def _test_single_initializer(self, i_param, inits, backend_config):
        # Given a set of initializer constructor arguments for the link, create
        # and initialize a link with those arguments. `i_param` holds the index
        # of the argument that should be tested among these.
        inits_orig = inits
        inits = [_get_initializer_argument_value(i) for i in inits]
        link, _, _ = self._create_initialized_link(inits, backend_config)

        # Extract the parameters from the initialized link.
        params = _get_link_params(link, self.param_names)

        # Convert the parameter of interest into a NumPy ndarray.
        cpu_device = backend.CpuDevice()
        param = params[i_param]
        param_xp = param.array
        param_np = cpu_device.send(param_xp)

        # The expected values of the parameter is decided by the given
        # initializer. If the initializer is `None`, it should have been
        # wrapped in a InitializerArgument along with the expected initializer
        # that the link should default to in case of `None`.
        #
        # Note that for this to work, the expected parameter must be inferred
        # deterministically.
        expected_init = _get_expected_initializer(inits_orig[i_param])
        expected_np = numpy.empty_like(param_np)
        expected_init(expected_np)

        # Compare the values of the expected and actual parameter.
        _check_arrays_equal(
            (expected_np,), (param_np,), LinkTestError,
            **self.check_initializers_options)


def _check_generated_initializer(init):
    if isinstance(init, InitializerArgument):
        init = init.expected_initializer
    elif init is None:
        raise ValueError(
            'A None initializer must be wrapped in a InitializerArgument '
            'along with the expected initializer fallen back to.')
    initializers._check_is_initializer_like(init)


def _get_initializer_argument_value(init):
    # Returns the initializer that should be passed to the link constructor.

    if isinstance(init, InitializerArgument):
        return init.argument_value
    return init


def _get_expected_initializer(init):
    # Returns the expected initializer for the given initializer.

    if isinstance(init, InitializerArgument):
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


def _check_array_types(arrays, device, func_name):
    if not isinstance(arrays, tuple):
        raise TypeError(
            '`{}()` must return a tuple, '
            'not {}.'.format(func_name, type(arrays)))
    if not all(
            a is None or isinstance(a, device.supported_array_types)
            for a in arrays):
        raise TypeError(
            '{}() must return a tuple of arrays supported by device {} or'
            ' None.\nActual: {}'.format(
                func_name, device, tuple([type(a) for a in arrays])))


def _check_variable_types(vars, device, func_name, test_error_cls):
    assert issubclass(test_error_cls, _TestError)

    if not isinstance(vars, tuple):
        test_error_cls.fail(
            '`{}()` must return a tuple, '
            'not {}.'.format(func_name, type(vars)))
    if not all(isinstance(a, chainer.Variable) for a in vars):
        test_error_cls.fail(
            '{}() must return a tuple of Variables.\n'
            'Actual: {}'.format(
                func_name, ', '.join(str(type(a)) for a in vars)))
    if not all(isinstance(a.array, device.supported_array_types)
               for a in vars):
        test_error_cls.fail(
            '{}() must return a tuple of Variables of arrays supported by '
            'device {}.\n'
            'Actual: {}'.format(
                func_name, device,
                ', '.join(str(type(a.array)) for a in vars)))


def _check_arrays_equal(
        actual_arrays, expected_arrays, test_error_cls, **opts):
    # `opts` is passed through to `testing.assert_all_close`.
    # Check all outputs are equal to expected values
    assert issubclass(test_error_cls, _TestError)

    message = None
    detail_message = None
    while True:
        # Check number of arrays
        if len(actual_arrays) != len(expected_arrays):
            message = (
                'Number of outputs ({}, {}) does not match'.format(
                    len(actual_arrays), len(expected_arrays)))
            break

        # Check dtypes and shapes
        dtypes_match = all([
            y.dtype == ye.dtype
            for y, ye in zip(actual_arrays, expected_arrays)])
        shapes_match = all([
            y.shape == ye.shape
            for y, ye in zip(actual_arrays, expected_arrays)])
        if not (shapes_match and dtypes_match):
            message = 'Shapes and/or dtypes do not match'
            break

        # Check values
        errors = []
        for i, (actual, expected) in (
                enumerate(zip(actual_arrays, expected_arrays))):
            try:
                array_module.assert_allclose(actual, expected, **opts)
            except AssertionError as e:
                errors.append((i, e))
        if errors:
            message = (
                'Outputs do not match the expected values.\n'
                'Indices of outputs that do not match: {}'.format(
                    ', '.join(str(i) for i, e in errors)))
            f = six.StringIO()
            for i, e in errors:
                f.write('Error details of output [{}]:\n'.format(i))
                f.write(str(e))
                f.write('\n')
            detail_message = f.getvalue()
            break
        break

    if message is not None:
        msg = (
            '{}\n'
            'Expected shapes and dtypes: {}\n'
            'Actual shapes and dtypes:   {}\n'.format(
                message,
                utils._format_array_props(expected_arrays),
                utils._format_array_props(actual_arrays)))
        if detail_message is not None:
            msg += '\n\n' + detail_message
        test_error_cls.fail(msg)
