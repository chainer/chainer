#pragma once

#include <memory>
#include <string>

#include <gsl/gsl>

#include "chainerx/backend.h"
#include "chainerx/device.h"
#include "chainerx/kernel_registry.h"

namespace chainerx {
namespace native {

class NativeDevice;
class NativeBackend;

namespace native_internal {

// Creates a device instance.
// This function is meant to be used from the backend class. Never use it for other purpose.
// This is defined in internal namespace in order to make it a friend of NativeDevice
// class.
NativeDevice* CreateDevice(NativeBackend& backend, int index);

}  // namespace native_internal

class NativeBackend : public Backend {
public:
    static constexpr const char* kDefaultName = "native";

    using Backend::Backend;

    std::string GetName() const override;

    int GetDeviceCount() const override;

    bool IsNative() const override { return true; }

    bool SupportsTransfer(Device& src_device, Device& dst_device) override;

    static KernelRegistry& GetGlobalKernelRegistry();

protected:
    KernelRegistry& GetParentKernelRegistry() override { return GetGlobalKernelRegistry(); }

private:
    std::unique_ptr<Device> CreateDevice(int index) override;
};

}  // namespace native
}  // namespace chainerx
