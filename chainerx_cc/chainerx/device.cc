#include "chainerx/device.h"

#include "chainerx/array.h"
#include "chainerx/context.h"
#include "chainerx/error.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/thread_local_state.h"

namespace chainerx {

void Device::CheckDevicesCompatible(const Array& array) {
    if (this != &array.device()) {
        throw DeviceError{"Device (", name(), ") is not compatible with array's device (", array.device().name(), ")."};
    }
}

namespace internal {

Device* GetDefaultDeviceNoExcept() noexcept { return internal::GetInternalThreadLocalState().default_device; }

}  // namespace internal

Device& GetDefaultDevice() {
    Device*& default_device = internal::GetInternalThreadLocalState().default_device;
    if (default_device == nullptr) {
        default_device = &GetDefaultContext().GetDevice({native::NativeBackend::kDefaultName, 0});
    }
    return *default_device;
}

void SetDefaultDevice(Device* device) {
    if (device != nullptr && &device->backend().context() != &GetDefaultContext()) {
        throw ContextError{"Context mismatch between default device and default context."};
    }
    Device*& default_device = internal::GetInternalThreadLocalState().default_device;
    default_device = device;
}

void CheckEqual(const Device& lhs, const Device& rhs) {
    if (&lhs != &rhs) {
        throw DeviceError{"Devices do not match: ", lhs.name(), ", ", rhs.name(), "."};
    }
}

}  // namespace chainerx
