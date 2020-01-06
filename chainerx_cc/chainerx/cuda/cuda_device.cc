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
    is_empty_ = false;
}

void MemoryKeeper::Collect() {
    // std::queue::empty() is not thread safe. Avoid using it in order to skip the lock.
    if (is_empty_) {
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
    is_empty_ = queue_.empty();
}

DeviceInternals& GetDeviceInternals(CudaDevice& device) { return device.device_internals_; }

}  // namespace cuda_internal

void CudaDevice::Synchronize() {
    CudaSetDeviceScope scope{index()};
    CheckCudaError(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace chainerx
