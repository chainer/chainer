import xchainer.testing  # NOQA

from xchainer._core import *  # NOQA

from builtins import bool, int, float  # NOQA

from xchainer import _core

from xchainer.creation.from_data import asanyarray  # NOQA
from xchainer.creation.from_data import fromfile  # NOQA
from xchainer.creation.from_data import fromfunction  # NOQA
from xchainer.creation.from_data import fromiter  # NOQA
from xchainer.creation.from_data import fromstring  # NOQA
from xchainer.creation.from_data import loadtxt  # NOQA

_global_context = _core.Context()
_core.set_global_default_context(_global_context)
