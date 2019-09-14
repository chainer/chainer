import inspect

from chainerx import _core
from chainerx._docs import array
from chainerx._docs import backend
from chainerx._docs import backprop
from chainerx._docs import context
from chainerx._docs import device
from chainerx._docs import routines
from chainerx._docs import utils


def set_doc(obj, docstring):
    if ((inspect.ismethoddescriptor(obj) or inspect.isroutine(obj))
            and not inspect.isfunction(obj)):
        # pybind-generated functions and methods
        _core._set_pybind_doc(obj, docstring)
        return

    obj.__doc__ = docstring


def set_docs():
    for m in (array, backend, backprop, context, device, routines, utils):
        m.set_docs()
