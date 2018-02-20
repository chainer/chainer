#include "xchainer/device_id.h"

#include "xchainer/error.h"

namespace xchainer {
namespace {

thread_local DeviceId thread_local_device_id = internal::kNullDeviceId;
static_assert(std::is_pod<decltype(thread_local_device_id)>::value, "thread_local_device_id must be POD");

}  // namespace

namespace internal {

const DeviceId& GetDefaultDeviceIdNoExcept() noexcept { return thread_local_device_id; }

}  // namespace internal

DeviceId::DeviceId(const std::string& name, Backend* backend) : name_(), backend_(backend) {
    if (name.size() >= device_id_detail::kMaxDeviceIdNameLength) {
        throw DeviceIdError("device_id name is too long; should be shorter than 8 characters");
    }
    std::copy(name.begin(), name.end(), static_cast<char*>(name_));
}

bool DeviceId::is_null() const { return *this == internal::kNullDeviceId; }

std::string DeviceId::ToString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const DeviceId& device_id) {
    if (device_id.is_null()) {
        os << "DeviceId(null)";
    } else {
        os << "DeviceId('" << device_id.name() << "')";
    }
    return os;
}

const DeviceId& GetDefaultDeviceId() {
    if (thread_local_device_id.is_null()) {
        throw XchainerError("Default device_id is not set.");
    } else {
        return thread_local_device_id;
    }
}

void SetDefaultDeviceId(const DeviceId& device_id) { thread_local_device_id = device_id; }

void DebugDumpDeviceId(std::ostream& os, const DeviceId& device_id) {
    if (device_id.is_null()) {
        os << "DeviceId(null)";
    } else {
        os << "DeviceId('" << device_id.name() << "', " << device_id.backend() << ")";
    }
}

}  // namespace xchainer
