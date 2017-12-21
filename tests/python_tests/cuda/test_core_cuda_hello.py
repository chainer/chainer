import pytest
import subprocess

import xchainer


cuda = pytest.mark.cuda


@cuda
def test_hello():
    proc = subprocess.Popen(['python', '-c',
                             'import xchainer; '
                             'xchainer.set_current_device("cpu"); '
                             'xchainer.hello()'], stdout=subprocess.PIPE)
    assert proc.stdout.read() == b'Hello, World!\n'

    proc = subprocess.Popen(['python', '-c',
                             'import xchainer; '
                             'xchainer.set_current_device("cuda"); '
                             'xchainer.hello()'], stdout=subprocess.PIPE)
    assert proc.stdout.read() == b'Hello, CUDA!\n'
