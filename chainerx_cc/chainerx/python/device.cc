#include "chainerx/python/device.h"

#include <memory>
#include <string>

#include "chainerx/backend.h"
#include "chainerx/context.h"
#include "chainerx/device.h"

#include "chainerx/python/common.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention

Device& GetDevice(py::handle handle) {
    if (handle.is_none()) {
        return GetDefaultDevice();
    }

    if (py::isinstance<Device&>(handle)) {
        return py::cast<Device&>(handle);
    }

    if (py::isinstance<py::str>(handle)) {
        // Device ID
        std::string device_id = py::cast<std::string>(handle);
        return GetDefaultContext().GetDevice(device_id);
    }

    throw py::type_error{"Device not understood: " + py::cast<std::string>(py::repr(handle))};
}

class PyDeviceScope {
public:
    explicit PyDeviceScope(Device& device) : device_{device} {}
    PyDeviceScope& Enter() {
        scope_ = std::make_unique<DeviceScope>(device_);
        return *this;
    }
    void Exit(py::args args) {
        (void)args;  // unused
        scope_.reset();
    }

    Device& device() const { return device_; }

private:
    // TODO(beam2d): better to replace it by "optional"...
    std::unique_ptr<DeviceScope> scope_;
    Device& device_;
};

void InitChainerxDevice(pybind11::module& m) {
    py::class_<Device> c{m, "Device"};
    c.def("__repr__", &Device::name);
    c.def("synchronize", &Device::Synchronize);
    c.def_property_readonly("name", &Device::name);
    c.def_property_readonly("backend", &Device::backend, py::return_value_policy::reference);
    c.def_property_readonly("context", &Device::context, py::return_value_policy::reference);
    c.def_property_readonly("index", &Device::index);

    m.def("get_default_device", []() -> Device& { return GetDefaultDevice(); }, py::return_value_policy::reference);
    m.def("set_default_device", [](Device& device) { SetDefaultDevice(&device); });
    m.def("set_default_device", [](const std::string& device_name) { SetDefaultDevice(&GetDefaultContext().GetDevice(device_name)); });
}

void InitChainerxDeviceScope(pybind11::module& m) {
    py::class_<PyDeviceScope> c{m, "DeviceScope"};
    c.def_property_readonly(
            "device", [](const PyDeviceScope& self) -> Device& { return self.device(); }, py::return_value_policy::reference);
    c.def("__enter__", &PyDeviceScope::Enter);
    c.def("__exit__", &PyDeviceScope::Exit);

    m.def("device_scope", [](Device& device) { return PyDeviceScope(device); });
    m.def("device_scope", [](const std::string& device_name) { return PyDeviceScope(GetDefaultContext().GetDevice(device_name)); });
    m.def("device_scope", [](const std::string& backend_name, int index) {
        return PyDeviceScope(GetDefaultContext().GetDevice({backend_name, index}));
    });
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
