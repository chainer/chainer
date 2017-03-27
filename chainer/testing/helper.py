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
    except pkg_resources.VersionConflict:
        skip = True

    msg = 'requires: {}'.format(','.join(requirements))
    return unittest.skipIf(skip, msg)


def _gen_supress_feature_warning_class(klass):
    assert issubclass(klass, unittest.TestCase)
    original_setup = klass.setUp
    original_teardown = klass.tearDown

    def setUp(self):
        self._warn = warnings.catch_warnings()
        self._warn.__enter__()
        warnings.simplefilter('ignore', FutureWarning)
        original_setup(self)

    def tearDown(self):
        self._warn.__exit__(None, None, None)
        original_teardown(self)

    klass.setUp = setUp
    klass.tearDown = tearDown

    return klass


def suppress_feature_warning():
    """Decorator that suppress feature warning.

    The feature warning is suppressed in decorated test class.

    .. admonition:: Example

       >>> import warnings
       ... from chainer import testing
       ...
       ... @testing.suppress_feature_warning()
       ... class Test(unittest.TestCase):
       ...     def test_warning(self):
       ...         warnings.warn('warn.', FutureWarning)

    """
    return _gen_supress_feature_warning_class
