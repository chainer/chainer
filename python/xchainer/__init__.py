from xchainer._core import *  # NOQA

from builtins import bool, int, float  # NOQA

from xchainer import _core


_global_context = _core.Context()
_core.set_global_default_context(_global_context)
