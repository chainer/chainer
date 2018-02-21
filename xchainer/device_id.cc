#include "xchainer/device_id.h"

#include "xchainer/backend.h"
#include "xchainer/error.h"

namespace xchainer {
namespace {

thread_local DeviceId thread_local_device_id = internal::kNullDeviceId;
static_assert(std::is_pod<decltype(thread_local_device_id)>::value, "thread_local_device_id must be POD");

}  // namespace

namespace internal {

const DeviceId& GetDefaultDeviceIdNoExcept() noexcept { return thread_local_device_id; }
}  // namespace internal

DeviceId::DeviceId(const std::string& device_name) {
    size_t pos = device_name.find(':');
    if (pos == std::string::npos) {
        backend_name_ = device_name;
        index_ = 0;
    } else {
        backend_name_ = device_name.substr(0, pos);
        try {
            // TODO(hvy): Check if device_name ends with the index without any garbage
            index_ = std::stoi(device_name.substr(pos + 1));
        } catch (const std::logic_error& e) {
            throw DeviceError("invalid device name (no integer found after ':'): '" + device_name + "'");
        }
        if (index_ < 0) {
            throw DeviceError("invalid device name (negative index is not allowed): '" + device_name + "'");
        }
    }
}

std::string DeviceId::ToString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const DeviceId& device_id) {
    os << backend_name_ << ':' << index_;
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

}  // namespace xchainer
