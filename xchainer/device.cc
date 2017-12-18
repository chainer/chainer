#include "xchainer/device.h"

#include <gsl/gsl>

#include "xchainer/error.h"

namespace xchainer {
namespace {
    thread_local Device thread_local_device = {"cpu"};
}

Device GetCurrentDevice() { return thread_local_device; }

void SetCurrentDevice(const Device& device) {
    if (device != Device{"cpu"} && device != Device{"cuda"}) {
        throw DeviceError("inavlid device");
    }
    thread_local_device = device;
}

}  // namespace xchainer
