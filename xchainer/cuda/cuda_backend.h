#pragma once

#include <memory>
#include <string>

#include <nonstd/optional.hpp>

#include "xchainer/backend.h"
#include "xchainer/context.h"
#include "xchainer/device.h"

namespace xchainer {
namespace cuda {

class CudaDevice;
class CudaBackend;

namespace cuda_internal {

// Creates a device instance.
// This function is meant to be used from the backend class. Never use it for other purpose.
// This is defined in cuda_internal namespace in order to make it a friend of CudaDevice
// class.
CudaDevice* CreateDevice(CudaBackend& backend, int index);

}  // namespace cuda_internal

class CudaBackend : public Backend {
public:
    static constexpr const char* kDefaultName = "cuda";
    static constexpr const size_t kCudnnDefaultMaxWorkspaceSize = 8 * 1024 * 1024;
    static constexpr const char* kCudnnMaxWorkspaceSizeEnvVarName = "XCHAINER_CUDNN_MAX_WORKSPACE_SIZE";

    using Backend::Backend;

    std::string GetName() const override;

    int GetDeviceCount() const override;

    bool SupportsTransfer(Device& src_device, Device& dst_device) override;

    void SetCudnnMaxWorkspaceSize(size_t max_workspace_size);

    size_t GetCudnnMaxWorkspaceSize();

private:
    std::unique_ptr<Device> CreateDevice(int index) override;
    nonstd::optional<size_t> cudnn_max_workspace_size_{};
    mutable std::mutex mutex_;
};

}  // namespace cuda
}  // namespace xchainer
