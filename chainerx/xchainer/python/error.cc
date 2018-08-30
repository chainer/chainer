#include "xchainer/python/error.h"

#include "xchainer/error.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention

void InitXchainerError(pybind11::module& m) {
    py::register_exception<XchainerError>(m, "XchainerError");
    py::register_exception<ContextError>(m, "ContextError");
    py::register_exception<BackendError>(m, "BackendError");
    py::register_exception<DeviceError>(m, "DeviceError");
    py::register_exception<DimensionError>(m, "DimensionError");
    py::register_exception<DtypeError>(m, "DtypeError");
    py::register_exception<NotImplementedError>(m, "NotImplementedError");
    py::register_exception<GradientCheckError>(m, "GradientCheckError");
}

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
