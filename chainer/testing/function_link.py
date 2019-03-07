import contextlib
import typing as tp  # NOQA
import unittest

import numpy

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


class InitializerNdarrayArgument(object):

    def __init__(self):
        self.generated = None

    def generate(self, shape, dtype):
        self.generated = numpy.random.uniform(-1, 1, shape).astype(dtype)
        return self.generated


class FunctionTestBase(object):
    backend_config = None
    check_forward_options = {}  # type: tp.Dict[str, tp.Any]
    check_backward_options = {}  # type: tp.Dict[str, tp.Any]
    check_double_backward_options = {}  # type: tp.Dict[str, tp.Any]
    skip_forward_test = False
    skip_backward_test = False
    skip_double_backward_test = False
    dodge_nondifferentiable = False
    contiguous = None

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
        self.before_test('test_forward')

        cpu_inputs = self._generate_inputs()
        inputs_copied = [a.copy() for a in cpu_inputs]

        # Compute expected outputs
        cpu_expected = self._forward_expected(cpu_inputs)
        inputs = backend_config.get_array(cpu_inputs)
        inputs = self._to_noncontiguous_as_needed(inputs)

        # Compute actual outputs
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

        if len(indices) > 0:
            FunctionTestError.fail(
                'Input arrays have been modified during forward.\n'
                'Indices of modified inputs: {}\n'
                'Input array shapes and dtypes: {}\n'.format(
                    ', '.join(str(i) for i in indices),
                    utils._format_array_props(inputs)))

        _check_forward_output_arrays_equal(
            cpu_expected,
            [var.array for var in outputs],
            'forward', FunctionTestError, **self.check_forward_options)

    def run_test_backward(self, backend_config):
        # Runs the backward test.

        if self.skip_backward_test:
            raise unittest.SkipTest('skip_backward_test is set')

        # avoid cyclic import
        from chainer import gradient_check

        self.backend_config = backend_config
        self.before_test('test_backward')

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
                    f, inputs, grad_outputs, dtype=numpy.float64,
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
        self.before_test('test_double_backward')

        def f(*args):
            return self._forward(args, backend_config)

        def do_check():
            inputs = self._generate_inputs()
            outputs = self._forward_expected(inputs)
            grad_outputs = self._generate_grad_outputs(outputs)
            grad_grad_inputs = self._generate_grad_grad_inputs(inputs)

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
                        dtype=numpy.float64,
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
        :class:`numpy.ndarray`.
        ``outputs_template`` is a tuple of template arrays. The returned arrays
        are expected to have the same shapes and dtypes as the template arrays.

    ``generate_grad_grad_inputs(self, inputs_template)``
        Returns a tuple of the second order input gradient arrays of type
        :class:`numpy.ndarray`.
        ``input_template`` is a tuple of template arrays. The returned arrays
        are expected to have the same shapes and dtypes as the template arrays.

    .. rubric:: Attributes

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

    ``contiguous`` (None or 'C'):
        Specifies the contiguousness of incoming arrays (i.e. inputs, output
        gradients, and the second order input gradients). If ``None``, the
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


class LinkTestCase(unittest.TestCase):

    """A base class for link test cases.

    Link test cases can inherit from this class to define a set of link tests.

    .. rubric:: Required methods

    Each concrete class must at least override the following methods.

    ``generate_inputs(self)``
        Returns a tuple of input arrays of type :class:`numpy.ndarray`.

    ``create_link(self, initializers)``
        Returns a link. The link should be initialized with the given
        initializer-likes ``initializers``. ``initializers`` is a tuple of
        same length as the number of parameters and contains initializer-likes
        returned by either ``generate_initializer_arguments`` or
        ``get_initializers`` depending on the test being run.

    .. rubric:: Optional methods

    Additionally, the concrete class can override the following methods. Some
    must be overridden depending on the skip flags  ``skip_forward_test``,
    ``skip_backward_test`` and ``skip_initializers_test``.

    ``generate_initializer_arguments(self)``
        Returns a tuple of initializers-likes. The tuple should contain an
        initializer-like for each initializer-like argument, i.e. the
        parameters to the link constructor. These will be passed to
        ``create_link`` for the forward and backward tests.
        This method must be implemented if either ``skip_forward_test`` or
        ``skip_backward_test`` is ``False`` in which case forward or backward
        tests are executed.

    ``get_initializers(self)``
        Returns a tuple with the same length as the number of initializers that
        the constructor of the link accepts. Each element in the tuple is a
        container itself, listing all initializers-likes that should be tested.
        Each initializer-like in the tuple is tested one at a time by being
        passed to ``create_link``. When the length of the tuple is greater than
        one (i.e. if the link accepts multiple initializers), the ones not
        being tested are replaced by the ones returned by
        `generate_initializer_arguments`.
        Initializer-likes returned here should be deterministic since test will
        invoke them multiple times to test the correctness.

        For testing initializer arguments that can be non-initializer values
        such as ``None``, one can use the ``InitializerArgument``, defining a
        pair of the link constructor argument and actual initializer-like used
        by the link.
        This method must be implemented if ``skip_initializers_test`` is
        ``False`` in which case the initializers test is executed.

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
        ``skip_backward_test`` is ``False`` in which case forward or backward
        tests are executed.

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

    ``param_names`` (list of str):
        A list of strings with all the names of the parameters that should be
        tested. E.g. ``['gamma', 'beta']`` for the batch normalization link.
        ``[]`` by default.

    ``dodge_nondifferentiable`` (bool):
        Enable non-differentiable point detection in numerical gradient
        calculation. If the data returned by
        ``generate_initializer_arguments``, ``create_link`` and
        ``generate_inputs`` turns out to be a non-differentiable point, the
        test will repeatedly resample those until a differentiable point will
        be finally sampled. ``False`` by default.

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

                def generate_initializer_arguments(self):
                    initialW = numpy.random.uniform(
                        -1, 1, (3, 2)).astype(numpy.float32)
                    initial_bias = numpy.random.uniform(
                        -1, 1, (3,)).astype(numpy.float32)
                    return initialW, initial_bias

                def get_initializers(self):
                    initialW = [initializers.Constant(1), 2]
                    initial_bias = [initializers.Constant(2), 3,
                        chainer.testing.link.InitializerArgument(None, 0)]
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
    check_param_initialization = True
    dodge_nondifferentiable = False
    contiguous = None

    # List of parameter names represented as strings.
    # I.e. ['gamma', 'beta'] for BatchNormalization.
    param_names = []

    def before_test(self, test_name):
        pass

    def generate_initializer_arguments(self):
        raise NotImplementedError(
            'generate_initializer_arguments is not implemented.')

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

        inits = self._generate_initializer_arguments()
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

        if self.check_param_initialization:
            for init, param_np in zip(inits, params_np):
                expected_init = _get_expected_initializer(init)
                if expected_init is not None:
                    if isinstance(expected_init, numpy.ndarray):
                        expected_np = expected_init
                    else:
                        expected_np = numpy.empty_like(param_np)
                        expected_init(expected_np)

                    # Compare the values of the expected and actual parameter.
                    _check_forward_output_arrays_equal(
                        expected_np, param_np, 'forward', LinkTestError,
                        **self.check_initializers_options)

        expected_outputs_np = self._forward_expected(inputs_np, params_np)

        _check_forward_output_arrays_equal(
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
            inits = self._generate_initializer_arguments()
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

    def _generate_initializer_arguments(self):
        params_init = self.generate_initializer_arguments()
        if not isinstance(params_init, (tuple, list)):
            raise TypeError(
                '`generate_initializer_arguments` must return a tuple or a '
                'list.')
        for init in params_init:
            # TODO(hvy): Check.
            # _check_generated_initializer(init)
            pass
        return params_init

    def _create_link(self, initializers, backend_config):
        link = self.create_link(initializers)
        if not isinstance(link, chainer.Link):
            raise TypeError(
                '`create_link` must return a chainer.Link object.')

        link.to_device(backend_config.device)

        return link

    def _create_initialized_link(
            self, inits, backend_config, return_inputs_outputs=False):
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

        _check_array_types(inputs, backend.CpuDevice(), 'generate_inputs')

        return inputs

    def _forward(self, link, inputs, backend_config):
        assert all(isinstance(x, chainer.Variable) for x in inputs)

        with backend_config:
            outputs = self.forward(link, inputs, backend_config.device)
        _check_variable_types(
            outputs, backend_config.device, 'forward', LinkTestError)

        return outputs

    def _forward_expected(self, inputs, params):
        assert all(isinstance(x, numpy.ndarray) for x in inputs)
        assert all(isinstance(x, numpy.ndarray) for x in params)

        outputs = self.forward_expected(inputs, params)
        _check_array_types(inputs, backend.CpuDevice(), 'test_forward')

        return outputs

    def _generate_grad_outputs(self, outputs_template):
        assert all(isinstance(x, numpy.ndarray) for x in outputs_template)

        grad_outputs = self.generate_grad_outputs(outputs_template)
        _check_array_types(
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
    if init is None:
        return

    initializers._check_is_initializer_like(init)


def _get_expected_initializer(init):
    # Returns the expected initializer for the given initializer.
    if init is None:
        return None

    if isinstance(init, InitializerNdarrayArgument):
        return init.generated

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
    if not all(isinstance(a, device.supported_array_types) for a in arrays):
        raise TypeError(
            '{}() must return a tuple of arrays supported by device {}.\n'
            'Actual: {}'.format(
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
                func_name, device, ', '.join(type(a.array) for a in vars)))


def _check_forward_output_arrays_equal(
        expected_arrays, actual_arrays, func_name, test_error_cls, **opts):
    # `opts` is passed through to `testing.assert_all_close`.
    # Check all outputs are equal to expected values
    assert issubclass(test_error_cls, _TestError)

    message = None
    while True:
        # Check number of arrays
        if len(expected_arrays) != len(actual_arrays):
            message = (
                'Number of outputs of {}() ({}, {}) does not '
                'match'.format(
                    func_name, len(expected_arrays), len(actual_arrays)))
            break

        # Check dtypes and shapes
        dtypes_match = all([
            ye.dtype == y.dtype
            for ye, y in zip(expected_arrays, actual_arrays)])
        shapes_match = all([
            ye.shape == y.shape
            for ye, y in zip(expected_arrays, actual_arrays)])
        if not (shapes_match and dtypes_match):
            message = (
                'Shapes and/or dtypes of {}() do not match'.format(func_name))
            break

        # Check values
        indices = []
        for i, (expected, actual) in (
                enumerate(zip(expected_arrays, actual_arrays))):
            try:
                array_module.assert_allclose(expected, actual, **opts)
            except AssertionError:
                indices.append(i)
        if len(indices) > 0:
            message = (
                'Outputs of {}() do not match the expected values.\n'
                'Indices of outputs that do not match: {}'.format(
                    func_name, ', '.join(str(i) for i in indices)))
            break
        break

    if message is not None:
        test_error_cls.fail(
            '{}\n'
            'Expected shapes and dtypes: {}\n'
            'Actual shapes and dtypes:   {}\n'.format(
                message,
                utils._format_array_props(expected_arrays),
                utils._format_array_props(actual_arrays)))
