#pragma once

#include <pybind11/pybind11.h>

#include "chainerx/macro.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

inline py::handle GetCachedNumpyModule() {
    static py::handle ret = py::module::import("numpy");
    return ret;
}

inline py::handle GetCachedNumpyArray() {
    static py::handle ret = py::module::import("numpy").attr("array");
    return ret;
}

inline py::handle GetCachedNumpyInteger() {
    static py::handle ret = py::module::import("numpy").attr("integer");
    return ret;
}

inline py::handle GetCachedNumpyNumber() {
    static py::handle ret = py::module::import("numpy").attr("number");
    return ret;
}

inline py::handle GetCachedNumpyBool() {
    static py::handle ret = py::module::import("numpy").attr("bool_");
    return ret;
}

inline py::handle GetCachedCupyModule() {
    static py::handle ret = py::module::import("cupy");
    return ret;
}

inline py::handle GetCachedCupyNdarray() {
    static py::handle ret = py::module::import("cupy").attr("ndarray");
    return ret;
}

inline py::handle GetCachedCupyMemoryPointer() {
    static py::handle ret = py::module::import("cupy").attr("cuda").attr("memory").attr("MemoryPointer");
    return ret;
}

inline py::handle GetCachedCupyUnownedMemory() {
    static py::handle ret = py::module::import("cupy").attr("cuda").attr("memory").attr("UnownedMemory");
    return ret;
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
