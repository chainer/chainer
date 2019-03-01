import contextlib
import typing as tp  # NOQA
import unittest

import numpy

import chainer
from chainer import backend
from chainer.testing import array as array_module
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
        _check_array_types(outputs, backend.CpuDevice(), 'forward_expected')
        return outputs

    def _forward(self, inputs, backend_config):
        assert all(isinstance(a, chainer.Variable) for a in inputs)
        with backend_config:
            outputs = self.forward(inputs, backend_config.device)
        _check_variable_types(outputs, backend_config.device, 'forward')
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
            'forward', **self.check_forward_options)

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
