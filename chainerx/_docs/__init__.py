import inspect

from chainerx import _core
from chainerx._docs import array
from chainerx._docs import context
from chainerx._docs import routines


def set_doc(obj, docstring):
    if inspect.ismethoddescriptor(obj) or inspect.isroutine(obj):
        # pybind-generated functions and methods
        _core._set_pybind_doc(obj, docstring)
        return

    obj.__doc__ = docstring


def set_docs():
    for m in (array, context, routines):
        m.set_docs()
