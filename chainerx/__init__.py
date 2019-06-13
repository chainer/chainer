import os
import sys


if sys.version_info[0] < 3:
    _available = False
else:
    try:
        from chainerx import _core
        _available = True
    except Exception:
        _available = False


if _available:
    from numpy import dtype  # NOQA
    from numpy import (
        bool_, int8, int16, int32, int64, uint8, float16, float32, float64)  # NOQA
    all_dtypes = (
        bool_, int8, int16, int32, int64, uint8, float16, float32, float64)

    from chainerx._core import *  # NOQA

    from builtins import bool, int, float  # NOQA

    from chainerx import _device  # NOQA

    from chainerx.creation.from_data import asanyarray  # NOQA
    from chainerx.creation.from_data import fromfile  # NOQA
    from chainerx.creation.from_data import fromfunction  # NOQA
    from chainerx.creation.from_data import fromiter  # NOQA
    from chainerx.creation.from_data import fromstring  # NOQA
    from chainerx.creation.from_data import loadtxt  # NOQA

    from chainerx.activation import relu  # NOQA

    from chainerx.manipulation.shape import ravel  # NOQA

    from chainerx.math.misc import clip  # NOQA

    from chainerx import random  # NOQA

    _global_context = _core.Context()
    _core.set_global_default_context(_global_context)

    # Implements ndarray methods in Python
    from chainerx import _ndarray
    _ndarray.populate()

    # Temporary workaround implementations that fall back to NumPy/CuPy's
    # respective functions.
    from chainerx import _fallback_workarounds
    _fallback_workarounds.populate()

    # Dynamically inject docstrings
    from chainerx import _docs
    _docs.set_docs()

    from chainerx import _cuda
    # Share memory pool with CuPy.
    if bool(int(os.getenv('CHAINERX_CUDA_CUPY_SHARE_ALLOCATOR', '0'))):
        _cuda.cupy_share_allocator()
else:
    class ndarray(object):

        """Dummy class for type testing."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError('chainerx is not available.')


def is_available():
    return _available
