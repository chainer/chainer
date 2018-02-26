#include "xchainer/python/backend.h"

#include "xchainer/backend.h"
#include "xchainer/device.h"

#include "xchainer/python/common.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

void InitXchainerBackend(pybind11::module& m) {
    py::class_<Backend> backend(m, "Backend");
    backend.def("get_device", &Backend::GetDevice, py::return_value_policy::reference).def_property_readonly("name", &Backend::GetName);
}

}  // namespace xchainer
