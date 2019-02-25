import contextlib

import chainer
from chainer.testing import array as array_module
from chainer import utils


class TestError(AssertionError):

    """Parent class to Chainer test errors.

    .. seealso::
        :class:`chainer.testing.FunctionTestError`
        :class:`chainer.testing.LinkTestError`

    """

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
    assert issubclass(test_error_cls, TestError)

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
        expected_arrays, actual_arrays, test_error_cls, **opts):
    # `opts` is passed through to `testing.assert_all_close`.
    # Check all outputs are equal to expected values
    assert issubclass(test_error_cls, TestError)

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
        test_error_cls.fail(
            '{}\n'
            'Expected shapes and dtypes: {}\n'
            'Actual shapes and dtypes:   {}\n'.format(
                message,
                utils._format_array_props(expected_arrays),
                utils._format_array_props(actual_arrays)))
