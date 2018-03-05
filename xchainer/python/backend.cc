#include "xchainer/python/backend.h"

#include "xchainer/backend.h"
#include "xchainer/context.h"
#include "xchainer/device.h"

#include "xchainer/python/common.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

void InitXchainerBackend(pybind11::module& m) {
    py::class_<Backend>(m, "Backend")
        .def("get_device", &Backend::GetDevice, py::return_value_policy::reference)
        .def_property_readonly("name", &Backend::GetName)
        .def_property_readonly("context", &Backend::context, py::return_value_policy::reference);
}

}  // namespace xchainer
