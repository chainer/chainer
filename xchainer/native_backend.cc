#include "xchainer/native_backend.h"

#include "xchainer/native_device.h"

namespace xchainer {

int NativeBackend::GetDeviceCount() const { return 4; }

Device& NativeBackend::GetDevice(int index) {
    if (!devices_.HasDevice(index)) {
        devices_.AddDevice(std::make_unique<NativeDevice>(*this, index));
    }
    return devices_.GetDevice(index);
}

std::string NativeBackend::GetName() const { return "native"; }

}  // namespace xchainer
