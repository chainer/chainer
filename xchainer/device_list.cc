#include "xchainer/device_list.h"

#include "xchainer/native_device.h"

namespace xchainer {

void DeviceList::AddDevice(std::unique_ptr<Device> device) {
    int index = device->index();
    if (index >= static_cast<int>(devices_.size())) {
        devices_.resize(index + 1);
    }
    devices_[index] = std::move(device);
}

Device& DeviceList::GetDevice(int index) const { return *(devices_.at(index).get()); }

bool DeviceList::HasDevice(int index) const {
    if (index >= static_cast<int>(devices_.size())) {
        return false;
    }
    return devices_[index].get() != nullptr;
}

}  // namespace xchainer
