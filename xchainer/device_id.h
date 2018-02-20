#pragma once

#include <cstring>
#include <sstream>
#include <string>

namespace xchainer {

namespace device_id_detail {

constexpr size_t kMaxDeviceIdNameLength = 8;

}  // device_id_detail

class Backend;

struct DeviceId {
public:
    DeviceId() = default;  // required to be POD
    DeviceId(const std::string& name, Backend* backend);

    std::string name() const { return name_; }
    Backend* backend() const { return backend_; }

    bool is_null() const;
    std::string ToString() const;

private:
    char name_[device_id_detail::kMaxDeviceIdNameLength];
    Backend* backend_;
};

namespace internal {

const DeviceId& GetDefaultDeviceIdNoExcept() noexcept;

constexpr DeviceId kNullDeviceId = {};

}  // namespace internal

inline bool operator==(const DeviceId& lhs, const DeviceId& rhs) { return (lhs.name() == rhs.name()) && (lhs.backend() == rhs.backend()); }

inline bool operator!=(const DeviceId& lhs, const DeviceId& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream&, const DeviceId&);

const DeviceId& GetDefaultDeviceId();

void SetDefaultDeviceId(const DeviceId& device_id);

// Scope object that switches the default device_id by RAII.
class DeviceIdScope {
public:
    DeviceIdScope() : orig_(internal::GetDefaultDeviceIdNoExcept()), exited_(false) {}
    explicit DeviceIdScope(DeviceId device_id) : DeviceIdScope() { SetDefaultDeviceId(device_id); }
    explicit DeviceIdScope(const std::string& device_id, Backend* backend) : DeviceIdScope(DeviceId{device_id, backend}) {}

    DeviceIdScope(const DeviceIdScope&) = delete;
    DeviceIdScope(DeviceIdScope&&) = delete;
    DeviceIdScope& operator=(const DeviceIdScope&) = delete;
    DeviceIdScope& operator=(DeviceIdScope&&) = delete;

    ~DeviceIdScope() { Exit(); }

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

void DebugDumpDeviceId(std::ostream& os, const DeviceId& device_id);

}  // namespace xchainer
