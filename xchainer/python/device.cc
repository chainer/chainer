#include "xchainer/python/device.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

#include "xchainer/backend.h"
#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/python/dtype.h"
#include "xchainer/python/shape.h"
#include "xchainer/python/strides.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace internal {

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
    explicit PyDeviceScope(Device& target) : target_(target) {}
    void Enter() { scope_ = std::make_unique<DeviceScope>(target_); }
    void Exit(py::args args) {
        (void)args;  // unused
        scope_.reset();
    }

private:
    // TODO(beam2d): better to replace it by "optional"...
    std::unique_ptr<DeviceScope> scope_;
    Device& target_;
};

// A device buffer that upon construction allocates device memory and creates a py::buffer_info, sharing ownership of the managed data
// (py::buffer_info only holds a raw pointer and does not manage the lifetime of the pointed data). Memoryviews created from this buffer
// will also share ownership. Note that accessing the .obj attribute of a memoryview may increase the reference count and should thus be
// avoided.
class PyDeviceBuffer {
public:
    PyDeviceBuffer(const std::shared_ptr<void>& data, const std::shared_ptr<py::buffer_info>& info) : data_{data}, info_{info} {}

    PyDeviceBuffer(
            const std::shared_ptr<void>& data,
            int64_t item_size,
            std::string format,
            int8_t ndim,
            const Shape& shape,
            const Strides& strides)
        : PyDeviceBuffer{data, std::make_shared<py::buffer_info>(data.get(), item_size, std::move(format), ndim, shape, strides)} {}

    std::shared_ptr<py::buffer_info> info() const { return info_; }

private:
    std::shared_ptr<void> data_;
    std::shared_ptr<py::buffer_info> info_;
};

void InitXchainerDevice(pybind11::module& m) {
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

void InitXchainerDeviceScope(pybind11::module& m) {
    py::class_<PyDeviceScope> c{m, "DeviceScope"};
    c.def("__enter__", &PyDeviceScope::Enter);
    c.def("__exit__", &PyDeviceScope::Exit);

    m.def("device_scope", [](Device& device) { return PyDeviceScope(device); });
    m.def("device_scope", [](const std::string& device_name) { return PyDeviceScope(GetDefaultContext().GetDevice(device_name)); });
    m.def("device_scope", [](const std::string& backend_name, int index) {
        return PyDeviceScope(GetDefaultContext().GetDevice({backend_name, index}));
    });
}

void InitXchainerDeviceBuffer(pybind11::module& m) {
    py::class_<PyDeviceBuffer> c{m, "DeviceBuffer", py::buffer_protocol()};
    c.def(py::init([](const py::list& list, const py::tuple& shape_tup, const py::handle& dtype_handle, const py::handle& device) {
              Shape shape = ToShape(shape_tup);
              int64_t total_size = shape.GetTotalSize();
              if (static_cast<size_t>(total_size) != list.size()) {
                  throw DimensionError{"Invalid data length"};
              }

              // Copy the Python list to a buffer on the host.
              Dtype dtype = GetDtype(dtype_handle);
              int64_t item_size = GetItemSize(dtype);
              int64_t bytes = item_size * total_size;
              std::shared_ptr<void> host_data = std::make_unique<uint8_t[]>(bytes);
              std::string format = VisitDtype(dtype, [&host_data, &list](auto pt) {
                  using T = typename decltype(pt)::type;
                  std::transform(list.begin(), list.end(), static_cast<T*>(host_data.get()), [](auto& item) { return py::cast<T>(item); });
                  return py::format_descriptor<T>::format();  // Return the dtype format, e.g. "f" for xchainer.float32.
              });

              // Copy the data on the host buffer to the target device.
              std::shared_ptr<void> device_data = internal::GetDevice(device).FromHostMemory(host_data, bytes);
              return PyDeviceBuffer{device_data, item_size, format, shape.ndim(), shape, Strides{shape, dtype}};
          }),
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("data"),
          py::arg("device") = nullptr);
    c.def_buffer([](const PyDeviceBuffer& self) {
        // py::buffer_info cannot be copied.
        std::shared_ptr<py::buffer_info> info = self.info();
        return py::buffer_info{info->ptr, info->itemsize, info->format, info->ndim, info->shape, info->strides};
    });
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
