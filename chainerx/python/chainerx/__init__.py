# `testing` needs to be imported before `_core`, because importing `_core` would populate sys.modules['chainerx.testing'].
from chainerx import testing  # NOQA

try:
    from chainerx._core import *  # NOQA
    _available = True
except ImportError:
    _available = False

    del testing


if _available:
    from builtins import bool, int, float  # NOQA

    from chainerx import _core

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
