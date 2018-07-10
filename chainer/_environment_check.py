import os
import sys


def _check_python_350():
    if sys.version_info[:3] == (3, 5, 0):
        if not int(os.getenv('CHAINER_PYTHON_350_FORCE', '0')):
            msg = """
    Chainer does not work with Python 3.5.0.

    We strongly recommend to use another version of Python.
    If you want to use Chainer with Python 3.5.0 at your own risk,
    set 1 to CHAINER_PYTHON_350_FORCE environment variable."""

            raise Exception(msg)


def check():
    _check_python_350()
