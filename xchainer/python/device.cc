#include "xchainer/python/device.h"

#include <sstream>

#include "xchainer/device.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

void InitXchainerDevice(pybind11::module& m) {
    py::class_<Device>(m, "Device")
        .def(py::init(&MakeDevice))
        .def("__eq__", py::overload_cast<const Device&, const Device&>(&operator==))
        .def("__ne__", py::overload_cast<const Device&, const Device&>(&operator!=))
        .def("__repr__",
             [](Device device) {
                 std::ostringstream os;
                 os << "<Device " << device.name << ">";
                 return os.str();
             });

    m.def("get_current_device", []() { return GetCurrentDevice(); });
    m.def("set_current_device", [](const Device& device) { SetCurrentDevice(device); });
    m.def("set_current_device", [](const std::string& name) { SetCurrentDevice(name); });
}

}  // namespace xchainer
