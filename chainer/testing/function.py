import contextlib
import os

import numpy

import chainer
from chainer import backend
from chainer.testing import array as array_module
from chainer.testing import condition
from chainer import utils


class FunctionTestError(AssertionError):
    """Raised when the target function is implemented incorrectly."""

    @staticmethod
    def check(expr, message):
        if not expr:
            raise FunctionTestError(message)

    @staticmethod
    def fail(message, exc=None):
        if exc is not None:
            utils._raise_from(FunctionTestError, message, exc)
        raise FunctionTestError(message)

    @staticmethod
    @contextlib.contextmanager
    def raise_if_fail(message, error_types=AssertionError):
        try:
            yield
        except error_types as e:
            FunctionTestError.fail(message, e)


def function_test():
    """A decorator to generate function tests.

    This decorator can be applied to a base :class:`unittest.TestCase` class
    to define a set of function tests in the class.

    .. rubric:: Required methods

    The base class must at least implement the following three methods.

    ``forward(inputs, device)``
        Implements the target forward function.
        ``inputs`` is a tuple of :class:`~chainer.Variable`\\ s.
        This method is expected to return the output
        :class:`~chainer.Variable`\\ s with the same array types as the inputs.
        ``device`` is the device corresponding to the input arrays.

    ``forward_expected(inputs)``
        Implements the expectation of the target forward function.
        ``inputs`` is a tuple of :class:`numpy.ndarray`\\ s.
        This method is expected to return the output
        :class:`numpy.ndarray`\\ s.

    ``generate_inputs()``
        Returns a tuple of input arrays of type :class:`numpy.ndarray`.

    .. rubric:: Optional methods

    Additionally the base class can implement the following methods.

    ``before_test(test_name)``
        A callback method called before each test.
        Typically a skip logic is implemented by conditionally raising
        :class:`unittest.SkipTest`.
        ``test_name`` is one of ``'test_forward'``, ``'test_backward'``, and
        ``'test_double_backward'``.

    ``generate_grad_outputs(outputs_template)``
        Returns a tuple of output gradient arrays of type
        :class:`numpy.ndarray`.
        ``outputs_template`` is a tuple of template arrays. The returned arrays
        are expected to have the same shapes and dtypes as the template arrays.

    ``generate_grad_grad_inputs(inputs_template)``
        Returns a tuple of the second order input gradient arrays of type
        :class:`numpy.ndarray`.
        ``input_template`` is a tuple of template arrays. The returned arrays
        are expected to have the same shapes and dtypes as the template arrays.

    .. rubric:: Class attributes

    The base class can define the following class attributes to control the
    behavior of the tests.

    ``forward_test`` (bool):
        Whether to test forward computation. ``True`` by default.

    ``backward_test`` (bool):
        Whether to test backward computation. ``True`` by default.

    ``double_backward_test`` (bool):
        Whether to test double-backward computation. ``True`` by default.

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

       This decorator assumes :func:`chainer.testing.inject_backend_tests`
       is used together. See the example below.

    .. admonition:: Example

       .. testcode::

          @chainer.testing.inject_backend_tests(
              None,
              [
                  {},  # CPU
                  {'use_cuda': True},  # GPU
              ])
          @chainer.testing.function_test()
          class TestReLU(unittest.TestCase):

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
    """

    def wrap(cls):
        # Assign class attributes
        attrs = [
            ('forward_test', True),
            ('backward_test', True),
            ('double_backward_test', True),
            ('dodge_nondifferentiable', False),
            ('contiguous', None),
        ]
        for attr_name, default_value in attrs:
            if not hasattr(cls, attr_name):
                setattr(cls, attr_name, default_value)

        # Replace __init__
        cls.__init__ = _FunctionTestTemplate.__dict__['__init__']

        # Enumerate member methods to copy
        method_names = []
        for memb_name in dir(_FunctionTestTemplate):
            obj = getattr(_FunctionTestTemplate, memb_name)
            if not hasattr(obj, '_copy_to_test_case'):
                continue
            enable_attr_name, = obj._copy_to_test_case
            if (enable_attr_name is not None
                    and not getattr(cls, enable_attr_name)):
                continue
            method_names.append(memb_name)

        # Copy member methods
        for method_name in method_names:
            if hasattr(cls, method_name):
                # Copy only if it's not defined in the concrete class
                continue
            setattr(
                cls,
                method_name,
                _FunctionTestTemplate.__dict__[method_name])

        # Configure test repeat count
        repeat_tests = [
            # (test method name, environment variable)
            ('test_forward', 'CHAINER_TEST_FORWARD_REPEAT'),
            ('test_backward', 'CHAINER_TEST_BACKWARD_REPEAT'),
            ('test_double_backward', 'CHAINER_TEST_DOUBLE_BACKWARD_REPEAT'),
        ]
        for method_name, repeat_var_name in repeat_tests:
            if not hasattr(cls, method_name):
                continue
            rep = int(os.environ.get(repeat_var_name, 1))
            if rep != 1:
                rep_method = condition.repeat(rep)(getattr(cls, method_name))
                setattr(cls, method_name, rep_method)

        cls._func_template_wrapped_class = cls
        return cls

    return wrap


def _copy_to_test_case(enable_attr_name=None):
    # Mark a method with this decorator to make it transferred to the concrete
    # test case class. If `enable_attr_name` is given, the transfer is skipped
    # if the class has this attribute and its value is False.
    def wrap(meth):
        meth._copy_to_test_case = (enable_attr_name,)
        return meth
    return wrap


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


def _check_variable_types(vars, device, func_name):
    if not isinstance(vars, tuple):
        FunctionTestError.fail(
            '`{}()` must return a tuple, '
            'not {}.'.format(func_name, type(vars)))
    if not all(isinstance(a, chainer.Variable) for a in vars):
        FunctionTestError.fail(
            '{}() must return a tuple of Variables.\n'
            'Actual: {}'.format(
                func_name, ', '.join(str(type(a)) for a in vars)))
    if not all(isinstance(a.array, device.supported_array_types)
               for a in vars):
        FunctionTestError.fail(
            '{}() must return a tuple of Variables of arrays supported by '
            'device {}.\n'
            'Actual: {}'.format(
                func_name, device, ', '.join(type(a.array) for a in vars)))


def _check_forward_output_arrays_equal(
        expected_arrays, actual_arrays, func_name, **opts):
    # `opts` is passed through to `testing.assert_all_close`.
    # Check all outputs are equal to expected values
    message = None
    while True:
        # Check number of arrays
        if len(expected_arrays) != len(actual_arrays):
            message = (
                'Number of outputs of forward() ({}, {}) does not '
                'match'.format(
                    len(expected_arrays), len(actual_arrays)))
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
                'Shapes and/or dtypes of forward() do not match'.format())
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
                'Outputs of forward() do not match the expected values.\n'
                'Indices of outputs that do not match: {}'.format(
                    ', '.join(str(i) for i in indices)))
            break
        break

    if message is not None:
        FunctionTestError.fail(
            '{}\n'
            'Expected shapes and dtypes: {}\n'
            'Actual shapes and dtypes:   {}\n'.format(
                message,
                utils._format_array_props(expected_arrays),
                utils._format_array_props(actual_arrays)))


class _FunctionTestTemplate(object):

    backend_config = None

    def __init__(self, *args, **kwargs):
        super(
            self._func_template_wrapped_class, self).__init__(*args, **kwargs)
        self.check_forward_options = {}
        self.check_backward_options = {}
        self.check_double_backward_options = {}

    @_copy_to_test_case()
    def before_test(self, test_name):
        pass

    @_copy_to_test_case()
    def forward(self, inputs, device):
        raise NotImplementedError('forward() is not implemented.')

    @_copy_to_test_case()
    def forward_expected(self, inputs):
        raise NotImplementedError('forward_expected() is not implemented.')

    @_copy_to_test_case()
    def generate_inputs(self):
        raise NotImplementedError('generate_inputs() is not implemented.')

    @_copy_to_test_case()
    def generate_grad_outputs(self, outputs_template):
        grad_outputs = tuple([
            numpy.random.uniform(-1, 1, a.shape).astype(a.dtype)
            for a in outputs_template])
        return grad_outputs

    @_copy_to_test_case()
    def generate_grad_grad_inputs(self, inputs_template):
        grad_grad_inputs = tuple([
            numpy.random.uniform(-1, 1, a.shape).astype(a.dtype)
            for a in inputs_template])
        return grad_grad_inputs

    @_copy_to_test_case()
    def _to_noncontiguous_as_needed(self, contig_arrays):
        if self.contiguous is None:
            # non-contiguous
            return array_module._as_noncontiguous_array(contig_arrays)
        if self.contiguous == 'C':
            # C-contiguous
            return contig_arrays
        assert False, (
            'Invalid value of `contiguous`: {}'.format(self.contiguous))

    @_copy_to_test_case()
    def _generate_inputs(self):
        inputs = self.generate_inputs()
        _check_array_types(inputs, backend.CpuDevice(), 'generate_inputs')
        return inputs

    @_copy_to_test_case()
    def _generate_grad_outputs(self, outputs_template):
        grad_outputs = self.generate_grad_outputs(outputs_template)
        _check_array_types(
            grad_outputs, backend.CpuDevice(), 'generate_grad_outputs')
        return grad_outputs

    @_copy_to_test_case()
    def _generate_grad_grad_inputs(self, inputs_template):
        grad_grad_inputs = self.generate_grad_grad_inputs(inputs_template)
        _check_array_types(
            grad_grad_inputs, backend.CpuDevice(), 'generate_grad_grad_inputs')
        return grad_grad_inputs

    @_copy_to_test_case()
    def _forward_expected(self, inputs):
        outputs = self.forward_expected(inputs)
        _check_array_types(outputs, backend.CpuDevice(), 'forward_expected')
        return outputs

    @_copy_to_test_case()
    def _forward(self, inputs, backend_config):
        assert all(isinstance(a, chainer.Variable) for a in inputs)
        with backend_config:
            outputs = self.forward(inputs, backend_config.device)
        _check_variable_types(outputs, backend_config.device, 'forward')
        return outputs

    @_copy_to_test_case(enable_attr_name='forward_test')
    def test_forward(self, backend_config):
        """Tests forward computation."""

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
            tuple([chainer.Variable(a) for a in inputs]),
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
            'forward', **self.check_forward_options)

    @_copy_to_test_case(enable_attr_name='backward_test')
    def test_backward(self, backend_config):
        """Tests backward computation."""

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

    @_copy_to_test_case(enable_attr_name='double_backward_test')
    def test_double_backward(self, backend_config):
        """Tests double-backward computation."""
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
