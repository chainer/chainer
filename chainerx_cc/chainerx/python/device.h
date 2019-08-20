#pragma once

#include <pybind11/pybind11.h>

#include "chainerx/device.h"

namespace chainerx {
namespace python {
namespace python_internal {

// The Python wrapper for chainerx::Device instances.
class PyDevice {
public:
    explicit PyDevice(Device& device) : device_{device} {}

    Device& device() const { return device_; }

    int index() const { return device_.index(); }

    Backend& backend() const { return device_.backend(); }
    Context& context() const { return device_.context(); }
    std::string name() const { return device_.name(); }

    void Synchronize() { device_.Synchronize(); }

private:
    Device& device_;
};

Device& GetDevice(pybind11::handle handle);

void InitChainerxDevice(pybind11::module& m);

void InitChainerxDeviceScope(pybind11::module& m);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
