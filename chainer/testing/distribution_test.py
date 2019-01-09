import unittest


try:
    import pytest  # NOQA
    _error = None
except ImportError as e:
    _error = e


if _error is None:
    from chainer.testing._distribution_test import distribution_unittest
else:
    class distribution_unittest(unittest.TestCase):
        def test_dummy(self):
            raise RuntimeError('''\
{} is not available.

Reason: {}: {}'''.format(__name__, type(_error).__name__, _error))
