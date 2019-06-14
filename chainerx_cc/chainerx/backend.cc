#include "chainerx/backend.h"

#include <string>
#include <utility>

#include "chainerx/device.h"
#include "chainerx/kernel_registry.h"

namespace chainerx {

Backend::~Backend() = default;

Backend::Backend(Context& context) : context_{context} {}

void Backend::Initialize() { kernel_registry_ = KernelRegistry{&GetParentKernelRegistry()}; }

Device& Backend::GetDevice(int index) {
    if (index < 0) {
        throw std::out_of_range{"The index number (= " + std::to_string(index) + ") is negative"};
    }
    std::unique_ptr<Device> device = CreateDevice(index);
    std::lock_guard<std::mutex> lock{devices_mutex_};
    if (devices_.size() <= static_cast<size_t>(index)) {
        devices_.resize(index + 1);
    }
    if (devices_[index] == nullptr) {
        devices_[index] = std::move(device);
    }
    return *devices_[index];
}

}  // namespace chainerx
