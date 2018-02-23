from xchainer._core import *  # NOQA

global_context = Context()
global_context.get_backend('native')
set_global_default_context(global_context)
