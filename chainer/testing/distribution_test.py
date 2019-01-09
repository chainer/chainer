import unittest

import chainer.types


try:
    import pytest  # NOQA
    _error = None
except ImportError as e:
    _error = e


if _error is None:
    from chainer.testing._distribution_test import distribution_unittest
elif not chainer.types.TYPE_CHECKING:
    class distribution_unittest(unittest.TestCase):
        def test_dummy(self):
            raise RuntimeError('''\
{} is not available.

Reason: {}: {}'''.format(__name__, type(_error).__name__, _error))
