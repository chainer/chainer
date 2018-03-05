#pragma once

#include <memory>
#include <string>

#include "xchainer/backend.h"
#include "xchainer/device.h"

namespace xchainer {
namespace cuda {

class CudaBackend : public Backend {
public:
    static constexpr const char* kDefaultName = "cuda";

    using Backend::Backend;

    std::string GetName() const override;

    int GetDeviceCount() const override;

    bool SupportsTransfer(Device& src_device, Device& dst_device) override;

private:
    std::unique_ptr<Device> CreateDevice(int index) override;
};

}  // namespace cuda
}  // namespace xchainer
