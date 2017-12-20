#include "xchainer/device.h"

#include "xchainer/error.h"

namespace xchainer {
namespace {

thread_local Device thread_local_device = {"cpu"};
}  // namespace

Device GetCurrentDevice() { return thread_local_device; }

void SetCurrentDevice(const Device& device) {
    if (device != Device{"cpu"} && device != Device{"cuda"}) {
        throw DeviceError("invalid device");
    }
    thread_local_device = device;
}

void SetCurrentDevice(const std::string& name) {
    Device device = {'\0'};
    if (name.size() >= sizeof(device.name)) {
        throw DeviceError("device name is too long; should be shorter than 8 characters");
    }
    std::copy(name.begin(), name.end(), device.name);
    SetCurrentDevice(device);
}

}  // namespace xchainer
