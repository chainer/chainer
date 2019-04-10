#include <pybind11/pybind11.h>

#include "chainerx/macro.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

inline py::handle& numpy_module() {
    static py::handle ret = py::module::import("numpy");
    return ret;
}

inline py::handle& numpy_array() {
    static py::handle ret = py::module::import("numpy").attr("array");
    return ret;
}

inline py::handle& numpy_integer() {
    static py::handle ret = py::module::import("numpy").attr("integer");
    return ret;
}

inline py::handle& cupy_module() {
    static py::handle ret = py::module::import("cupy");
    return ret;
}

inline py::handle& cupy_ndarray() {
    static py::handle ret = py::module::import("cupy").attr("ndarray");
    return ret;
}

inline py::handle& cupy_cuda_memory_MemoryPointer() {
    static py::handle ret = py::module::import("cupy").attr("cuda").attr("memory").attr("MemoryPointer");
    return ret;
}

inline py::handle& cupy_cuda_memory_UnownedMemory() {
    static py::handle ret = py::module::import("cupy").attr("cuda").attr("memory").attr("UnownedMemory");
    return ret;
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
