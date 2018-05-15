#include "xchainer/python/testing/testing_module.h"

#include "xchainer/python/testing/device_buffer.h"

#include <pybind11/pybind11.h>

namespace xchainer {
namespace python {
namespace testing {
namespace internal {

void InitXchainerTestingModule(pybind11::module& m) { InitXchainerDeviceBuffer(m); }

}  // namespace internal
}  // namespace testing
}  // namespace python
}  // namespace xchainer
