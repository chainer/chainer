import unittest

import six

import chainer
from chainer import _runtime_info


class TestRuntimeInfo(unittest.TestCase):
    def test_get_runtime_info(self):
        info = _runtime_info.get_runtime_info()
        assert chainer.__version__ in str(info)

    def test_print_runtime_info(self):
        out = six.StringIO()
        _runtime_info.print_runtime_info(out)
        assert out.getvalue() == str(_runtime_info.get_runtime_info())
