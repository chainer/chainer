#include "xchainer/device.h"

#include <type_traits>

#include "xchainer/array.h"
#include "xchainer/context.h"
#include "xchainer/error.h"
#include "xchainer/native_backend.h"

namespace xchainer {
namespace {

thread_local Device* t_default_device = nullptr;
static_assert(std::is_pod<decltype(t_default_device)>::value, "t_default_device must be POD");

}  // namespace

void Device::CheckDevicesCompatible(const Array& array) {
    if (this != &array.device()) {
        throw DeviceError("Device (" + name() + ") is not compatible with array's device (" + array.device().name() + ").");
    }
}

namespace internal {

Device* GetDefaultDeviceNoExcept() noexcept { return t_default_device; }

}  // namespace internal

Device& GetDefaultDevice() {
    if (t_default_device == nullptr) {
        t_default_device = &GetDefaultContext().GetDevice({NativeBackend::kDefaultName, 0});
    }
    return *t_default_device;
}

void SetDefaultDevice(Device* device) {
    if (device != nullptr && &device->backend().context() != &GetDefaultContext()) {
        throw ContextError("Context mismatch between default device and default context.");
    }
    t_default_device = device;
}

}  // namespace xchainer
