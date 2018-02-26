from xchainer._core import *  # NOQA

_global_context = Context()
_global_context.get_backend('native')
set_global_default_context(_global_context)
