#include "chainerx/python/common_export.h"

#include "chainerx/python/testing/device_buffer.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "chainerx/device.h"
#include "chainerx/error.h"
#include "chainerx/python/device.h"
#include "chainerx/python/dtype.h"
#include "chainerx/python/shape.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

#include "chainerx/python/common.h"

namespace chainerx {
namespace python {
namespace testing {
namespace testing_internal {

namespace py = pybind11;  // standard convention
using py::literals::operator""_a;

// A device buffer that upon construction allocates device memory and creates a py::buffer_info, sharing ownership of the managed data
// (py::buffer_info only holds a raw pointer and does not manage the lifetime of the pointed data). Memoryviews created from this buffer
// will also share ownership. Note that accessing the .obj attribute of a memoryview may increase the reference count and should thus be
// avoided.
class PyDeviceBuffer {
public:
    PyDeviceBuffer(std::shared_ptr<void> data, std::shared_ptr<py::buffer_info> info) : data_{std::move(data)}, info_{std::move(info)} {}

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

void InitChainerxDeviceBuffer(pybind11::module& m) {
    py::class_<PyDeviceBuffer> c{m, "_DeviceBuffer", py::buffer_protocol()};
    c.def(py::init([](const py::list& list, py::handle shape_tup, const py::handle& dtype_handle, const py::handle& device) {
              Shape shape = python_internal::ToShape(shape_tup);
              int64_t total_size = shape.GetTotalSize();
              if (static_cast<size_t>(total_size) != list.size()) {
                  throw DimensionError{"Invalid data length"};
              }

              // Copy the Python list to a buffer on the host.
              Dtype dtype = python_internal::GetDtype(dtype_handle);
              int64_t item_size = GetItemSize(dtype);
              int64_t bytes = item_size * total_size;
              std::shared_ptr<void> host_data{new uint8_t[bytes], std::default_delete<uint8_t[]>()};
              std::string format = VisitDtype(dtype, [&host_data, &list](auto pt) {
                  using T = typename decltype(pt)::type;
                  std::transform(list.begin(), list.end(), static_cast<T*>(host_data.get()), [](auto& item) { return py::cast<T>(item); });
                  return py::format_descriptor<T>::format();  // Return the dtype format, e.g. "f" for chainerx.float32.
              });

              // Copy the data on the host buffer to the target device.
              std::shared_ptr<void> device_data = python_internal::GetDevice(device).FromHostMemory(host_data, bytes);
              return PyDeviceBuffer{device_data, item_size, format, shape.ndim(), shape, Strides{shape, dtype}};
          }),
          "shape"_a,
          "dtype"_a,
          "data"_a,
          "device"_a = nullptr);
    c.def_buffer([](const PyDeviceBuffer& self) {
        // py::buffer_info cannot be copied.
        std::shared_ptr<py::buffer_info> info = self.info();
        return py::buffer_info{info->ptr, info->itemsize, info->format, info->ndim, info->shape, info->strides};
    });
}

}  // namespace testing_internal
}  // namespace testing
}  // namespace python
}  // namespace chainerx
