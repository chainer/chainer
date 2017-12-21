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

}  // namespace xchainer
