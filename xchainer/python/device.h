#pragma once

#include <string>

#include <pybind11/pybind11.h>
#include <nonstd/optional.hpp>

#include "xchainer/context.h"
#include "xchainer/device.h"

namespace xchainer {
namespace python {
namespace internal {

inline Device& GetDevice(const nonstd::optional<std::string>& device_id) {
    return device_id.has_value() ? GetDefaultContext().GetDevice(device_id.value()) : GetDefaultDevice();
}

void InitXchainerDevice(pybind11::module&);

void InitXchainerDeviceScope(pybind11::module&);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
