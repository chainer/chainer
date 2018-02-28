#include "xchainer/memory.h"

#include <cassert>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA

#include "xchainer/backend.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/error.h"

namespace xchainer {
namespace internal {

bool IsPointerCudaMemory(const void* ptr) {
#ifdef XCHAINER_ENABLE_CUDA
    cudaPointerAttributes attr = {};
    cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
    switch (status) {
        case cudaSuccess:
            if (attr.isManaged) {
                return true;
            } else {
                throw XchainerError("Non-managed GPU memory is not supported");
            }
        case cudaErrorInvalidValue:
            return false;
        default:
            cuda::CheckError(status);
            break;
    }
    assert(false);  // should never be reached
#else
    (void)ptr;  // unused
    return false;
#endif  // XCHAINER_ENABLE_CUDA
}

std::shared_ptr<void> Allocate(Device& device, size_t bytesize) { return device.Allocate(bytesize); }

void MemoryCopy(Device& device, void* dst_ptr, const void* src_ptr, size_t bytesize) { device.MemoryCopyFrom(dst_ptr, src_ptr, bytesize, device); }

std::shared_ptr<void> MemoryFromBuffer(Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    return device.FromBuffer(src_ptr, bytesize);
}

}  // namespace internal
}  // namespace xchainer
