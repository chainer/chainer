#pragma once

#include <memory>
#include <string>

#include <absl/types/optional.h>
#include <gsl/gsl>

#include "chainerx/backend.h"
#include "chainerx/context.h"
#include "chainerx/device.h"
#include "chainerx/kernel_registry.h"

namespace chainerx {
namespace cuda {

class CudaDevice;
class CudaBackend;

namespace cuda_internal {

// Creates a device instance.
// This function is meant to be used from the backend class. Never use it for other purpose.
// This is defined in cuda_internal namespace in order to make it a friend of CudaDevice
// class.
gsl::owner<CudaDevice*> CreateDevice(CudaBackend& backend, int index);

}  // namespace cuda_internal

class CudaBackend : public Backend {
public:
    static constexpr const char* kDefaultName = "cuda";
    static constexpr const size_t kCudnnDefaultMaxWorkspaceSize = 8 * 1024 * 1024;
    static constexpr const char* kCudnnMaxWorkspaceSizeEnvVarName = "CHAINERX_CUDNN_MAX_WORKSPACE_SIZE";

    using Backend::Backend;

    std::string GetName() const override;

    int GetDeviceCount() const override;

    bool SupportsTransfer(Device& src_device, Device& dst_device) override;

    // TODO(hvy): Move to CudaDevice.
    // Sets maximum cuDNN workspace size.
    // This value is shared across threads.
    void SetCudnnMaxWorkspaceSize(size_t max_workspace_size);

    // TODO(hvy): Move to CudaDevice.
    // Gets maximum cuDNN workspace size.
    size_t GetCudnnMaxWorkspaceSize();

    static KernelRegistry& GetGlobalKernelRegistry();

protected:
    KernelRegistry& GetParentKernelRegistry() override { return GetGlobalKernelRegistry(); }

private:
    std::unique_ptr<Device> CreateDevice(int index) override;

    // TODO(hvy): Move to CudaDevice.
    absl::optional<size_t> cudnn_max_workspace_size_{};

    std::mutex mutex_;
};

}  // namespace cuda
}  // namespace chainerx
