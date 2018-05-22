#pragma once

#include <memory>
#include <string>

#include "xchainer/backend.h"
#include "xchainer/context.h"
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

    void SetMaxWorkspaceSize(size_t max_workspace_size) { max_workspace_size_ = max_workspace_size; }
    size_t max_workspace_size() { return max_workspace_size_; }

private:
    std::unique_ptr<Device> CreateDevice(int index) override;
    size_t max_workspace_size_ = 8 * 1024 * 1024;
};

}  // namespace cuda
}  // namespace xchainer
