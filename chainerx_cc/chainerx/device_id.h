#pragma once

#include <cstring>
#include <sstream>
#include <string>
#include <utility>

namespace chainerx {

class Backend;

class DeviceId {
public:
    DeviceId(const std::string& device_name);  // NOLINT
    DeviceId(std::string backend_name, int index) : backend_name_{std::move(backend_name)}, index_{index} {}

    const std::string& backend_name() const { return backend_name_; }
    int index() const { return index_; }

    std::string ToString() const;

private:
    std::string backend_name_;
    int index_;
};

inline bool operator==(const DeviceId& lhs, const DeviceId& rhs) {
    return lhs.backend_name() == rhs.backend_name() && lhs.index() == rhs.index();
}

inline bool operator!=(const DeviceId& lhs, const DeviceId& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream& os, const DeviceId& device_id);

}  // namespace chainerx
