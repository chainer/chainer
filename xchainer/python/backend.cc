#include "xchainer/python/backend.h"

#include "xchainer/backend.h"
#include "xchainer/context.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;  // standard convention

void InitXchainerBackend(pybind11::module& m) {
    py::class_<Backend> cls{m, "Backend"};
    cls.def("get_device", &Backend::GetDevice, py::return_value_policy::reference);
    cls.def("get_device_count", &Backend::GetDeviceCount);
    cls.def_property_readonly("name", &Backend::GetName);
    cls.def_property_readonly("context", &Backend::context, py::return_value_policy::reference);
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
