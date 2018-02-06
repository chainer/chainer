#pragma once

#include <cstring>
#include <string>

namespace xchainer {

class Backend;

constexpr size_t kMaxDeviceNameLength = 8;

struct Device {
public:
    const std::string name() const { return name_; }

    const Backend* backend() const { return backend_; }

    static Device MakeDevice(const std::string& name, Backend* backend);

private:
    char name_[kMaxDeviceNameLength];
    Backend* backend_;
};

constexpr Device kNullDevice = {};

namespace internal {

const Device& GetCurrentDeviceNoExcept() noexcept;

}  // namespace internal

// TODO(sonots): Need to prohibit to create devices of the same name?
inline bool operator==(const Device& lhs, const Device& rhs) { return (lhs.name() == rhs.name()) && (lhs.backend() == rhs.backend()); }

inline bool operator!=(const Device& lhs, const Device& rhs) { return !(lhs == rhs); }

const Device& GetCurrentDevice();

void SetCurrentDevice(const Device& device);

void SetCurrentDevice(const std::string& name, Backend* backend);

// Scope object that switches the current device by RAII.
class DeviceScope {
public:
    DeviceScope() : orig_(internal::GetCurrentDeviceNoExcept()) {}
    explicit DeviceScope(Device device) : DeviceScope() { SetCurrentDevice(device); }
    explicit DeviceScope(const std::string& device, Backend* backend) : DeviceScope(Device::MakeDevice(device, backend)) {}

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope(DeviceScope&&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    DeviceScope& operator=(DeviceScope&&) = delete;

    ~DeviceScope() { Exit(); }

    // Explicitly recovers the original device. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (orig_ != kNullDevice) {
            SetCurrentDevice(orig_);
        }
        orig_ = kNullDevice;
    }

private:
    Device orig_;
};

}  // namespace xchainer
