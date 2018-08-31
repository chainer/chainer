#include "chainerx/python/error.h"

#include "chainerx/error.h"

#include "chainerx/python/common.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention

void InitChainerxError(pybind11::module& m) {
    py::register_exception<ChainerxError>(m, "ChainerxError");
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
}  // namespace chainerx
