import unittest

import packaging.version

import chainer
from chainer import testing


class TestVersion(unittest.TestCase):
    def test_pep440(self):
        # raises InvalidVersion unless a valid PEP 440 version is given
        packaging.version.Version(chainer.__version__)


testing.run_module(__name__, __file__)
