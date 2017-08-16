import contextlib
import pkg_resources
import unittest
import warnings


def with_requires(*requirements):
    """Run a test case only when given requirements are satisfied.

    .. admonition:: Example

       This test case runs only when `numpy>=1.10` is installed.

       >>> from chainer import testing
       ... class Test(unittest.TestCase):
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


@contextlib.contextmanager
def assert_warns(expected):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        yield

    if any(isinstance(m.message, expected) for m in w):
        return

    try:
        exc_name = expected.__name__
    except AttributeError:
        exc_name = str(expected)

    raise AssertionError('%s not triggerred' % exc_name)
