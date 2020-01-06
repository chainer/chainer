#include "chainerx/cuda/cuda_backend.h"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <absl/types/optional.h>
#include <cuda_runtime.h>
#include <gsl/gsl>

#include "chainerx/backend.h"
#include "chainerx/context.h"
#include "chainerx/cuda/cuda_device.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/util.h"

namespace chainerx {
namespace cuda {

constexpr const char* CudaBackend::kDefaultName;
constexpr const size_t CudaBackend::kCudnnDefaultMaxWorkspaceSize;
constexpr const char* CudaBackend::kCudnnMaxWorkspaceSizeEnvVarName;

namespace cuda_internal {

gsl::owner<CudaDevice*> CreateDevice(CudaBackend& backend, int index) { return new CudaDevice{backend, index}; }

}  // namespace cuda_internal

std::string CudaBackend::GetName() const { return kDefaultName; }

int CudaBackend::GetDeviceCount() const {
    int count = 0;
    CheckCudaError(cudaGetDeviceCount(&count));
    return count;
}

std::unique_ptr<Device> CudaBackend::CreateDevice(int index) {
    int device_count = GetDeviceCount();
    if (index >= device_count) {
        throw std::out_of_range{"The index number (= " + std::to_string(index) +
                                ") is not less than the device count (= " + std::to_string(device_count) + ')'};
    }
    return std::unique_ptr<CudaDevice>(cuda_internal::CreateDevice(*this, index));
}

bool CudaBackend::SupportsTransfer(Device& src_device, Device& dst_device) {
    Backend& src_backend = src_device.backend();
    Backend& dst_backend = dst_device.backend();
    if (&src_backend == this) {
        return &dst_backend == this || dst_backend.IsNative();
    }
    if (&dst_backend == this) {
        return &src_backend == this || src_backend.IsNative();
    }
    return false;
}

void CudaBackend::SetCudnnMaxWorkspaceSize(size_t max_workspace_size) {
    std::lock_guard<std::mutex> lock{mutex_};
    cudnn_max_workspace_size_ = max_workspace_size;
}

size_t CudaBackend::GetCudnnMaxWorkspaceSize() {
    std::lock_guard<std::mutex> lock{mutex_};
    if (cudnn_max_workspace_size_) {
        return *cudnn_max_workspace_size_;
    }
    if (absl::optional<std::string> env = GetEnv(kCudnnMaxWorkspaceSizeEnvVarName)) {
        cudnn_max_workspace_size_ = std::stoul(*env);
    } else {
        cudnn_max_workspace_size_ = kCudnnDefaultMaxWorkspaceSize;
    }
    return *cudnn_max_workspace_size_;
}

KernelRegistry& CudaBackend::GetGlobalKernelRegistry() {
    static gsl::owner<KernelRegistry*> global_kernel_registry = new KernelRegistry{};
    return *global_kernel_registry;
}

}  // namespace cuda
}  // namespace chainerx
