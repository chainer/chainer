#include "xchainer/python/device.h"

#include <memory>
#include <sstream>

#include "xchainer/backend.h"
#include "xchainer/device.h"

#include "xchainer/python/common.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

class PyDeviceScope {
public:
    explicit PyDeviceScope(Device target) : target_(target) {}
    void Enter() { scope_ = std::make_unique<DeviceScope>(target_); }
    void Exit(py::args args) {
        (void)args;  // unused
        scope_.reset();
    }

private:
    // TODO(beam2d): better to replace it by "optional"...
    std::unique_ptr<DeviceScope> scope_;
    Device target_;
};

void InitXchainerDevice(pybind11::module& m) {
    py::class_<Device>(m, "Device")
        .def(py::init(&MakeDevice))
        .def("__eq__", py::overload_cast<const Device&, const Device&>(&operator==))
        .def("__ne__", py::overload_cast<const Device&, const Device&>(&operator!=))
        .def("__repr__",
             [](Device device) {
                 std::ostringstream os;
                 os << "<Device('" << static_cast<char*>(device.name) << "', " << device.backend << ")>";
                 return os.str();
             })
        .def_property_readonly("name", [](const Device& self) { return self.name; })
        .def_property_readonly("backend", [](const Device& self) { return self.backend; });

    m.def("get_current_device", []() { return GetCurrentDevice(); });
    m.def("set_current_device", [](const Device& device) { SetCurrentDevice(device); });
    // TODO(sonots): Get device or backend from global dict and set to current device
    // m.def("set_current_device", [](const std::string& name) { SetCurrentDevice(name); });

    py::class_<PyDeviceScope>(m, "DeviceScope").def("__enter__", &PyDeviceScope::Enter).def("__exit__", &PyDeviceScope::Exit);
    // TODO(sonots): Get device or backend from global dict and set to current device
    // m.def("device_scope", [](const std::string& name) { return PyDeviceScope(MakeDevice(name)); });
    m.def("device_scope", [](Device device) { return PyDeviceScope(device); });
}

}  // namespace xchainer
