#include "xchainer/device_id.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "xchainer/error.h"

namespace xchainer {

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

}  // namespace xchainer
