import contextlib
import pkg_resources
import sys
import unittest
import warnings

try:
    import mock
    _mock_error = None
except ImportError as e:
    _mock_error = e


def _check_mock_available():
    if _mock_error is not None:
        raise RuntimeError(
            'mock is not available: Reason: {}'.format(_mock_error))


def with_requires(*requirements):
    """Run a test case only when given requirements are satisfied.

    .. admonition:: Example

       This test case runs only when `numpy>=1.10` is installed.

       >>> import unittest
       >>> from chainer import testing
       >>> class Test(unittest.TestCase):
       ...     @testing.with_requires('numpy>=1.10')
       ...     def test_for_numpy_1_10(self):
       ...         pass

    Args:
        requirements: A list of string representing requirement condition to
            run a given test case.

    """
    ws = pkg_resources.WorkingSet()
    try:
        ws.require(*requirements)
        skip = False
    except pkg_resources.ResolutionError:
        skip = True

    msg = 'requires: {}'.format(','.join(requirements))
    return unittest.skipIf(skip, msg)


def without_requires(*requirements):
    """Run a test case only when given requirements are not satisfied.

    .. admonition:: Example

    This test case runs only when `numpy>=1.10` is not installed.

    >>> from chainer import testing
    ... class Test(unittest.TestCase):
    ...     @testing.without_requires('numpy>=1.10')
    ...     def test_without_numpy_1_10(self):
    ...         pass

    Args:
    requirements: A list of string representing requirement condition to
        run a given test case.

    """
    ws = pkg_resources.WorkingSet()
    try:
        ws.require(*requirements)
        skip = True
    except pkg_resources.ResolutionError:
        skip = False

    msg = 'requires: {}'.format(','.join(requirements))
    return unittest.skipIf(skip, msg)


@contextlib.contextmanager
def assert_warns(expected):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        yield

    # Python 2 does not raise warnings multiple times from the same stack
    # frame.
    if sys.version_info >= (3, 0):
        if not any(isinstance(m.message, expected) for m in w):
            try:
                exc_name = expected.__name__
            except AttributeError:
                exc_name = str(expected)

            raise AssertionError('%s not triggerred' % exc_name)


def _import_object_from_name(fullname):
    comps = fullname.split('.')
    obj = sys.modules.get(comps[0])
    if obj is None:
        raise RuntimeError('Can\'t import {}'.format(comps[0]))
    for i, comp in enumerate(comps[1:]):
        obj = getattr(obj, comp)
        if obj is None:
            raise RuntimeError(
                'Can\'t find object {}'.format('.'.join(comps[:i + 1])))
    return obj


def patch(target, *args, **kwargs):
    """A wrapper of mock.patch which appends wraps argument.

    .. note::

       Unbound methods are not supported as ``wraps`` argument.

    Args:
        target(str): Full name of target object.
        wraps: Wrapping object which will be passed to ``mock.patch`` as
            ``wraps`` argument.
            If omitted, the object specified by ``target`` is used.
        *args: Passed to ``mock.patch``.
        **kwargs: Passed to ``mock.patch``.

    """
    _check_mock_available()
    try:
        wraps = kwargs.pop('wraps')
    except KeyError:
        wraps = _import_object_from_name(target)
    return mock.patch(target, *args, wraps=wraps, **kwargs)
