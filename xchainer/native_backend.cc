#include "xchainer/native_backend.h"

#include <stdexcept>
#include <string>

#include "xchainer/native_device.h"

namespace xchainer {

std::string NativeBackend::GetName() const { return kDefaultName; }

// TODO(sonots): Returns number of CPU cores
int NativeBackend::GetDeviceCount() const { return 4; }

std::unique_ptr<Device> NativeBackend::CreateDevice(int index) {
    int device_count = GetDeviceCount();
    if (index >= device_count) {
        throw std::out_of_range("The index number (= " + std::to_string(index) + ") is not less than the device count (= " +
                                std::to_string(device_count) + ')');
    }
    return std::make_unique<NativeDevice>(*this, index);
}

}  // namespace xchainer
