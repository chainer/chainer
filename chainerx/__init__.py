import sys


if sys.version_info[0] < 3:
    _available = False
else:
    # `testing` needs to be imported before `_core`, because importing `_core` would populate sys.modules['chainerx.testing'].
    from chainerx import testing  # NOQA

    try:
        from chainerx import _core
        _available = True
    except Exception:
        del testing
        _available = False


if _available:
    from numpy import dtype, bool_, int8, int16, int32, int64, uint8, float32, float64  # NOQA

    from chainerx._core import *  # NOQA

    from builtins import bool, int, float  # NOQA

    from chainerx.creation.from_data import asanyarray  # NOQA
    from chainerx.creation.from_data import fromfile  # NOQA
    from chainerx.creation.from_data import fromfunction  # NOQA
    from chainerx.creation.from_data import fromiter  # NOQA
    from chainerx.creation.from_data import fromstring  # NOQA
    from chainerx.creation.from_data import loadtxt  # NOQA

    _global_context = _core.Context()
    _core.set_global_default_context(_global_context)


def is_available():
    return _available
