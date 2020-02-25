#include "chainerx/python/common_export.h"

#include "chainerx/python/backend.h"

#include "chainerx/backend.h"
#include "chainerx/context.h"

#include "chainerx/python/common.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention

void InitChainerxBackend(pybind11::module& m) {
    py::class_<Backend> c{m, "Backend"};
    c.def("get_device", &Backend::GetDevice, py::return_value_policy::reference);
    c.def("get_device_count", &Backend::GetDeviceCount);
    c.def_property_readonly("name", &Backend::GetName);
    c.def_property_readonly("context", &Backend::context, py::return_value_policy::reference);
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
