import unittest

import six

from chainer import testing
from chainer.utils.argument import parse_kwargs


class TestArgument(unittest.TestCase):

    def test_parse_kwargs(self):

        def test(**kwargs):
            return parse_kwargs(kwargs, ('foo', 1), ('bar', 2))

        self.assertEqual(test(), (1, 2))
        self.assertEqual(test(bar=1, foo=2), (2, 1))

        re = r'test\(\) got unexpected keyword argument\(s\) \'ham\', \'spam\''
        with six.assertRaisesRegex(self, TypeError, re):
            test(spam=1, ham=2)


testing.run_module(__name__, __file__)
