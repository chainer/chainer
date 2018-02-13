#include "xchainer/memory.h"

#include <cassert>
#include <cstring>

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

void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) {
#ifdef XCHAINER_ENABLE_CUDA
    bool is_dst_cuda_memory = IsPointerCudaMemory(dst_ptr);
    bool is_src_cuda_memory = IsPointerCudaMemory(src_ptr);
    if (is_dst_cuda_memory) {
        if (is_src_cuda_memory) {
            // Copy from device to device is faster even in unified memory
            cuda::CheckError(cudaMemcpy(dst_ptr, src_ptr, bytesize, cudaMemcpyDeviceToDevice));
        } else {
            // For pre-6.x GPU architecture, we encountered SEGV with std::memcpy
            // ref. https://github.com/pfnet/xchainer/pull/74
            cuda::CheckError(cudaMemcpy(dst_ptr, src_ptr, bytesize, cudaMemcpyHostToDevice));
        }
    } else {
        if (is_src_cuda_memory) {
            // For pre-6.x GPU architecture, we encountered SEGV with std::memcpy
            // ref. https://github.com/pfnet/xchainer/pull/74
            cuda::CheckError(cudaMemcpy(dst_ptr, src_ptr, bytesize, cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(dst_ptr, src_ptr, bytesize);
        }
    }
#else
    std::memcpy(dst_ptr, src_ptr, bytesize);
#endif  // XCHAINER_ENABLE_CUDA
}

std::shared_ptr<void> MemoryFromBuffer(Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize) {
// TODO(sonots): Use device.FromBuffer()
#ifdef XCHAINER_ENABLE_CUDA
    if (device.backend().GetName() == "native") {
        if (IsPointerCudaMemory(src_ptr.get())) {
            std::shared_ptr<void> dst_ptr = Allocate(device, bytesize);
            cuda::CheckError(cudaMemcpy(dst_ptr.get(), src_ptr.get(), bytesize, cudaMemcpyDeviceToHost));
            return dst_ptr;
        } else {
            return src_ptr;
        }
    } else if (device.backend().GetName() == "cuda") {
        if (IsPointerCudaMemory(src_ptr.get())) {
            return src_ptr;
        } else {
            std::shared_ptr<void> dst_ptr = Allocate(device, bytesize);
            cuda::CheckError(cudaMemcpy(dst_ptr.get(), src_ptr.get(), bytesize, cudaMemcpyHostToDevice));
            return dst_ptr;
        }
    } else {
        throw DeviceError("invalid device");
    }
#else
    (void)bytesize;  // unused
    if (device.backend().GetName() == "native") {
        return src_ptr;
    } else {
        throw DeviceError("invalid device");
    }
#endif  // XCHAINER_ENABLE_CUDA
}

}  // namespace internal
}  // namespace xchainer
