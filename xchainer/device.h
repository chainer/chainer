#pragma once

#include <string>

namespace xchainer {

struct Device {
    std::string name;
};

inline bool operator==(Device lhs, Device rhs) { return lhs.name == rhs.name; }

inline bool operator!=(Device lhs, Device rhs) { return !(lhs == rhs); }

Device GetCurrentDevice();

void SetCurrentDevice(const Device& device);

void SetCurrentDevice(const std::string& device) { SetCurrentDevice(Device{device}); };

}  // namespace xchainer
