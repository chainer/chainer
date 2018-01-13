#include "xchainer/python/device.h"

#include <memory>
#include <sstream>

#include "xchainer/device.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

class PyDeviceScope {
public:
    explicit PyDeviceScope(Device target) : target_(target) {}
    void Enter() { scope_ = std::make_unique<DeviceScope>(target_); }
    void Exit(py::args args) {
        (void)args;
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
        .def("__repr__", [](Device device) {
            std::ostringstream os;
            os << "<Device " << static_cast<char*>(device.name) << ">";
            return os.str();
        });

    m.def("get_current_device", []() { return GetCurrentDevice(); });
    m.def("set_current_device", [](const Device& device) { SetCurrentDevice(device); });
    m.def("set_current_device", [](const std::string& name) { SetCurrentDevice(name); });

    py::class_<PyDeviceScope>(m, "DeviceScope").def("__enter__", &PyDeviceScope::Enter).def("__exit__", &PyDeviceScope::Exit);
    m.def("device_scope", [](const std::string& device) { return PyDeviceScope(MakeDevice(device)); });
    m.def("device_scope", [](Device device) { return PyDeviceScope(device); });
}

}  // namespace xchainer
