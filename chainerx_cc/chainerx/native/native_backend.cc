#include "chainerx/native/native_backend.h"

#include <memory>
#include <stdexcept>
#include <string>

#include <gsl/gsl>

#include "chainerx/native/native_device.h"

namespace chainerx {
namespace native {

constexpr const char* NativeBackend::kDefaultName;

namespace native_internal {

gsl::owner<NativeDevice*> CreateDevice(NativeBackend& backend, int index) { return new NativeDevice{backend, index}; }

}  // namespace native_internal

std::string NativeBackend::GetName() const { return kDefaultName; }

// TODO(sonots): Returns number of CPU cores
int NativeBackend::GetDeviceCount() const { return 4; }

std::unique_ptr<Device> NativeBackend::CreateDevice(int index) {
    int device_count = GetDeviceCount();
    if (index >= device_count) {
        throw std::out_of_range{"The index number (= " + std::to_string(index) +
                                ") is not less than the device count (= " + std::to_string(device_count) + ')'};
    }
    return std::unique_ptr<NativeDevice>(native_internal::CreateDevice(*this, index));
}

bool NativeBackend::SupportsTransfer(Device& src_device, Device& dst_device) {
    return &src_device.backend() == this && &dst_device.backend() == this;
}

KernelRegistry& NativeBackend::GetGlobalKernelRegistry() {
    static gsl::owner<KernelRegistry*> global_kernel_registry = new KernelRegistry{};
    return *global_kernel_registry;
}

}  // namespace native
}  // namespace chainerx
