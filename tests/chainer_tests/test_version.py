import unittest
import packaging.version

import chainer


class TestVersion(unittest.TestCase):
    def test_pep440(self):
        version = packaging.version.parse(chainer.__version__)
        self.assertIsInstance(version, packaging.version.Version)


testing.run_module(__name__, __file__)
