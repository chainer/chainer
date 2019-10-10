#pragma once

#include <pybind11/pybind11.h>

#include "chainerx/device.h"

namespace chainerx {
namespace python {
namespace python_internal {

// TODO(take-cheeze): Add API that can inherit the source array's device for chainerx/python/routines.cc
Device& GetDevice(pybind11::handle handle);

void InitChainerxDevice(pybind11::module& m);

void InitChainerxDeviceScope(pybind11::module& m);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
