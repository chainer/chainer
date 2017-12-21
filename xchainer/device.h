#pragma once

#include <cstring>
#include <string>

namespace xchainer {

struct Device {
    char name[8];
};

inline bool operator==(const Device& lhs, const Device& rhs) { return strncmp(lhs.name, rhs.name, 8) == 0; }

inline bool operator!=(const Device& lhs, const Device& rhs) { return !(lhs == rhs); }

Device MakeDevice(const std::string& name);

Device GetCurrentDevice();

void SetCurrentDevice(const Device& device);

void SetCurrentDevice(const std::string& name);

// Scope object that switches the current device by RAII.
class DeviceScope {
public:
    DeviceScope() : orig_(GetCurrentDevice()) {}
    explicit DeviceScope(Device device) : DeviceScope() { SetCurrentDevice(device); }
    explicit DeviceScope(const std::string& device) : DeviceScope(MakeDevice(device)) {}

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope(DeviceScope&&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    DeviceScope& operator=(DeviceScope&&) = delete;

    ~DeviceScope() { Exit(); }

    // Explicitly recovers the original device. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (orig_ != Device{}) {
            SetCurrentDevice(orig_);
        }
        orig_ = Device{};
    }

private:
    Device orig_;
};

}  // namespace xchainer
