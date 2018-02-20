#include "xchainer/native_backend.h"

#include <stdexcept>

#include "xchainer/native_device.h"

namespace xchainer {

// TODO(sonots): Returns number of CPU cores
int NativeBackend::GetDeviceCount() const { return 4; }

Device& NativeBackend::GetDevice(int index) {
    if (index < 0) {
        throw std::out_of_range("The index number must be greater than or equal to 0");
    }
    if (index >= GetDeviceCount()) {
        throw std::out_of_range("The index number must be smaller than the number of available devices");
    }
    if (!devices_.HasDevice(index)) {
        devices_.AddDevice(std::make_unique<NativeDevice>(*this, index));
    }
    return devices_.GetDevice(index);
}

std::string NativeBackend::GetName() const { return "native"; }

}  // namespace xchainer
