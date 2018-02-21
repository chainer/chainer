#pragma once

#include <cstring>
#include <sstream>
#include <string>
#include <utility>

namespace xchainer {

class Backend;

class DeviceId {
public:
    DeviceId(const std::string& device_name);
    DeviceId(std::string backend_name, int index) : backend_name_(std::move(backend_name)), index_(index) {}

    const std::string& backend_name() const { return backend_name_; }
    int index() const { return index_; }

    std::string ToString() const;

private:
    std::string backend_name_;
    int index_;
};

namespace internal {

Device* GetDefaultDeviceNoExcept() noexcept;

}  // namespace internal

inline bool operator==(const DeviceId& lhs, const DeviceId& rhs) { return lhs.backend() == rhs.backend() && lhs.index() == rhs.index(); }

inline bool operator!=(const DeviceId& lhs, const DeviceId& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream&, const DeviceId&);

Device& GetDefaultDevice();

void SetDefaultDevice(Device* device);

// Scope object that switches the default device_id by RAII.
class DeviceScope {
public:
    DeviceScope() : orig_(internal::GetDefaultDeviceNoExcept()), exited_(false) {}
    explicit DeviceScope(Device& device) : DeviceScope() { SetDefaultDevice(&device); }

    // TODO(hvy): Maybe unnecessary.
    explicit DeviceScope(Backend* backend, int index = 0) : DeviceScope(backend->GetDevice(index)) {}

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope(DeviceScope&&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    DeviceScope& operator=(DeviceScope&&) = delete;

    ~DeviceScope() { Exit(); }

    // Explicitly recovers the original device. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (!exited_) {
            SetDefaultDevice(orig_);
            exited_ = true;
        }
    }

private:
    Device* orig_;
    bool exited_;
};

}  // namespace xchainer
