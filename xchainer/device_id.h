#pragma once

#include <cstring>
#include <sstream>
#include <string>

namespace xchainer {

class Backend;

// TODO(hvy): Follow struct coding guidelines.
// TODO(hvy): Replace backend pointer to backend name.
struct DeviceId {
public:
    DeviceId() = default;  // required to be POD
    DeviceId(Backend* backend, int index = 0) : backend_(backend), index_(index) {}

    Backend* backend() const { return backend_; }
    int index() const { return index_; }

    bool is_null() const;
    std::string ToString() const;

private:
    Backend* backend_;
    int index_;
};

namespace internal {

const DeviceId& GetDefaultDeviceIdNoExcept() noexcept;

constexpr DeviceId kNullDeviceId = {};

}  // namespace internal

inline bool operator==(const DeviceId& lhs, const DeviceId& rhs) { return lhs.backend() == rhs.backend() && lhs.index() == rhs.index(); }

inline bool operator!=(const DeviceId& lhs, const DeviceId& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream&, const DeviceId&);

const DeviceId& GetDefaultDeviceId();

void SetDefaultDeviceId(const DeviceId& device_id);

// Scope object that switches the default device_id by RAII.
class DeviceScope {
public:
    DeviceScope() : orig_(internal::GetDefaultDeviceIdNoExcept()), exited_(false) {}
    explicit DeviceScope(DeviceId device_id) : DeviceScope() { SetDefaultDeviceId(device_id); }
    explicit DeviceScope(Backend* backend, int index = 0) : DeviceScope(DeviceId{backend, index}) {}

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope(DeviceScope&&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    DeviceScope& operator=(DeviceScope&&) = delete;

    ~DeviceScope() { Exit(); }

    // Explicitly recovers the original device_id. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (!exited_) {
            SetDefaultDeviceId(orig_);
            exited_ = true;
        }
    }

private:
    DeviceId orig_;
    bool exited_;
};

}  // namespace xchainer
