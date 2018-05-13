#pragma once

#include <string>

#include <pybind11/pybind11.h>
#include <nonstd/optional.hpp>

#include "xchainer/context.h"
#include "xchainer/device.h"

namespace xchainer {
namespace python {
namespace internal {

Device& GetDevice(pybind11::handle handle);

void InitXchainerDevice(pybind11::module&);

void InitXchainerDeviceScope(pybind11::module&);

void InitXchainerDeviceBuffer(pybind11::module&);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
