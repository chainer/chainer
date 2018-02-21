#include "xchainer/python/device_id.h"

#include <memory>
#include <sstream>

#include "xchainer/backend.h"
#include "xchainer/device_id.h"

#include "xchainer/python/common.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

class PyDeviceScope {
public:
    explicit PyDeviceScope(DeviceId target) : target_(target) {}
    void Enter() { scope_ = std::make_unique<DeviceScope>(target_); }
    void Exit(py::args args) {
        (void)args;  // unused
        scope_.reset();
    }

private:
    // TODO(beam2d): better to replace it by "optional"...
    std::unique_ptr<DeviceScope> scope_;
    DeviceId target_;
};

void InitXchainerDeviceId(pybind11::module& m) {
    py::class_<DeviceId>(m, "DeviceId")
        // We uses py::keep_alive because device_id keeps the pointer to the backend through its lifetime
        .def(py::init<Backend*, int>(), py::keep_alive<1, 2>(), py::arg(), py::arg("index") = 0)
        .def("__eq__", py::overload_cast<const DeviceId&, const DeviceId&>(&operator==))
        .def("__ne__", py::overload_cast<const DeviceId&, const DeviceId&>(&operator!=))
        .def("__repr__", &DeviceId::ToString)
        .def_property_readonly("backend", &DeviceId::backend)
        .def_property_readonly("index", &DeviceId::index);

    m.def("get_default_device_id", []() { return GetDefaultDeviceId(); });
    m.def("set_default_device_id", [](const DeviceId& device_id) { SetDefaultDeviceId(device_id); });

    py::class_<PyDeviceScope>(m, "DeviceScope").def("__enter__", &PyDeviceScope::Enter).def("__exit__", &PyDeviceScope::Exit);
    m.def("device_scope", [](DeviceId device_id) { return PyDeviceScope(device_id); });
}

}  // namespace xchainer
