#pragma once

#include <memory>
#include <string>
#include <vector>

#include "xchainer/device.h"

namespace xchainer {

class DeviceList {
public:
    void AddDevice(std::unique_ptr<Device> device);

    // Caller must guarantee the Device of given index is already added
    Device& GetDevice(int index) const;

    bool HasDevice(int index) const;

private:
    std::vector<std::unique_ptr<Device>> devices_;
};

}  // namespace xchainer
