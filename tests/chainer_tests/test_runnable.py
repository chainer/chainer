import io
import os
import re
import unittest

from chainer import testing


class TestRunnable(unittest.TestCase):

    def test_runnable(self):
        cwd = os.path.dirname(__file__)
        regex = re.compile(r'^test_.*\.py$')
        for dirpath, dirnames, filenames in os.walk(cwd):
            for filename in filenames:
                if not regex.match(filename):
                    continue
                path = os.path.join(dirpath, filename)
                with io.open(path, encoding='utf-8') as f:
                    source = f.read()
                self.assertIn('testing.run_module(__name__, __file__)',
                              source,
                              '''{0} is not runnable.
Call testing.run_module at the end of the test.'''.format(path))


testing.run_module(__name__, __file__)
