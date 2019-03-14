#include "chainerx/cuda/cuda_device.h"

#include <memory>
#include <mutex>
#include <utility>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"

namespace chainerx {
namespace cuda {
namespace cuda_internal {

MemoryKeeper::~MemoryKeeper() {
    while (!queue_.empty()) {
        const std::pair<cudaEvent_t, std::shared_ptr<void>>& pair = queue_.front();
        cudaEventDestroy(pair.first);
        queue_.pop();
    }
}

void MemoryKeeper::Add(cudaStream_t stream, std::shared_ptr<void> memory) {
    // TODO(niboshi): Currently only the default stream is supported.
    CHAINERX_ASSERT(stream == nullptr);

    cudaEvent_t event{};
    CheckCudaError(cudaEventCreate(&event));
    CheckCudaError(cudaEventRecord(event, stream));

    std::lock_guard<std::mutex> lock{mutex_};
    queue_.emplace(event, std::move(memory));
}

void MemoryKeeper::Collect() {
    if (queue_.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock{mutex_};

    while (true) {
        if (queue_.empty()) {
            break;
        }
        std::pair<cudaEvent_t, std::shared_ptr<void>>& pair = queue_.front();
        if (cudaSuccess != cudaEventQuery(pair.first)) {
            break;
        }

        CheckCudaError(cudaEventDestroy(pair.first));
        queue_.pop();
    }
}

DeviceInternals& GetDeviceInternals(CudaDevice& device) { return device.device_internals_; }

}  // namespace cuda_internal

CudaDevice::CudaDevice(CudaBackend& backend, int index)
    : Device{backend, index},
      device_memory_pool_{std::make_shared<MemoryPool>(index, std::make_unique<DeviceMemoryAllocator>())},
      pinned_memory_pool_{std::make_shared<MemoryPool>(index, std::make_unique<PinnedMemoryAllocator>())},
      device_internals_{index} {
    cudaDeviceProp device_prop{};
    CheckCudaError(cudaGetDeviceProperties(&device_prop, index));
    compute_capability_ = ComputeCapability{device_prop.major, device_prop.minor};
}

void CudaDevice::Synchronize() {
    CudaSetDeviceScope scope{index()};
    CheckCudaError(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace chainerx
