#include "xchainer/device.h"

#include "xchainer/error.h"

namespace xchainer {
namespace {

thread_local Device thread_local_device = {"cpu"};
// Device must be POD (plain old data) to be used as a thread local variable safely.
// ref. https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables
static_assert(std::is_pod<decltype(thread_local_device)>::value, "thread_local_device must be POD");

}  // namespace

Device MakeDevice(const std::string& name) {
    Device device = {};
    if (name.size() >= sizeof(device.name)) {
        throw DeviceError("device name is too long; should be shorter than 8 characters");
    }
    std::copy(name.begin(), name.end(), static_cast<char*>(device.name));
    return device;
}

Device GetCurrentDevice() { return thread_local_device; }

void SetCurrentDevice(const Device& device) {
    if (device != Device{"cpu"} && device != Device{"cuda"}) {
        throw DeviceError("invalid device");
    }
    thread_local_device = device;
}

void SetCurrentDevice(const std::string& name) {
    auto device = MakeDevice(name);
    SetCurrentDevice(device);
}

}  // namespace xchainer
