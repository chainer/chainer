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
    // TODO(takagi): continue mob from here
}

std::string DeviceId::ToString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const DeviceId& device_id) {
    if (device_id.is_null()) {
        os << "DeviceId(null)";
    } else {
        os << "DeviceId('" << device_id.backend()->GetName() << "', " << device_id.index() << ")";
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

}  // namespace xchainer
