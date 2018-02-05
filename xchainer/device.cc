#include "xchainer/device.h"

#include "xchainer/error.h"

namespace xchainer {
namespace {

thread_local Device thread_local_device = kDefaultDevice;
// Device must be POD (plain old data) to be used as a thread local variable safely.
// ref. https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables
static_assert(std::is_pod<decltype(thread_local_device)>::value, "thread_local_device must be POD");

}  // namespace

namespace internal {

Device GetCurrentDeviceNoExcept() noexcept { return thread_local_device; }

}  // namespace internal

Device MakeDevice(const std::string& name, Backend* backend) {
    Device device = {};
    if (name.size() >= kMaxDeviceNameLength) {
        throw DeviceError("device name is too long; should be shorter than 8 characters");
    }
    std::copy(name.begin(), name.end(), static_cast<char*>(device.name));
    device.backend = backend;
    return device;
}

Device GetCurrentDevice() {
    Device device = thread_local_device;
    if (device == kDefaultDevice) {
        throw XchainerError("No device is available. Please set via SetCurrentDevice()");
    } else {
        return device;
    }
}

void SetCurrentDevice(const Device& device) { thread_local_device = device; }

void SetCurrentDevice(const std::string& name, Backend* backend) {
    auto device = MakeDevice(name, backend);
    SetCurrentDevice(device);
}

}  // namespace xchainer
