#pragma once

#include <pybind11/pybind11.h>

#include "chainerx/device.h"

namespace chainerx {
namespace python {
namespace python_internal {

Device& GetDevice(pybind11::handle handle);

void InitXchainerDevice(pybind11::module&);

void InitXchainerDeviceScope(pybind11::module&);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
