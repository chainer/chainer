#include "xchainer/python/testing/testing_module.h"

#include "xchainer/python/testing/device_buffer.h"

#include <pybind11/pybind11.h>

#include "xchainer/python/array.h"
#include "xchainer/python/device.h"

namespace xchainer {
namespace python {
namespace testing {
namespace internal {

namespace py = pybind11;

void InitXchainerTestingModule(pybind11::module& m) {
    InitXchainerDeviceBuffer(m);

    // Converts from NumPy array. It supports keepstrides option.
    m.def("_fromnumpy",
          [](py::array array, bool keepstrides, py::handle device) {
              if (keepstrides) {
                  return xchainer::python::internal::MakeArrayFromNumpyArray(array, xchainer::python::internal::GetDevice(device));
              }
              return xchainer::python::internal::MakeArray(array, array.dtype(), true, device);
          },
          py::arg("array"),
          py::arg("keepstrides") = false,
          py::arg("device") = nullptr);
}

}  // namespace internal
}  // namespace testing
}  // namespace python
}  // namespace xchainer
