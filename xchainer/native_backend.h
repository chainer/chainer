#pragma once

#include <memory>
#include <string>
#include <tuple>

#include "xchainer/backend.h"
#include "xchainer/device.h"

namespace xchainer {

class NativeBackend : public Backend {
public:
    static constexpr const char* kDefaultName = "native";

    using Backend::Backend;

    std::string GetName() const override;

    int GetDeviceCount() const override;

    bool SupportsTransfer(Device& src_device, Device& dst_device) override;

private:
    std::unique_ptr<Device> CreateDevice(int index) override;
};

}  // namespace xchainer
