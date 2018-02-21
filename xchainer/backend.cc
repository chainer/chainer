#include "xchainer/backend.h"

#include "xchainer/device.h"

namespace xchainer {

Device& Backend::GetDevice(int index) {
    if (index < 0) {
        throw std::out_of_range("The index number must be greater than or equal to 0");
    }
    if (index >= GetDeviceCount()) {
        throw std::out_of_range("The index number must be smaller than the number of available devices");
    }
    if (devices_.size() <= static_cast<size_t>(index)) {
        devices_.resize(index + 1);
    }
    if (devices_[index] == nullptr) {
        devices_[index] = CreateDevice(index);
    }
    return *devices_[index];
}

}  // namespace xchainer
