#include "xchainer/memory.h"

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA

#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/error.h"

namespace xchainer {

bool IsPointerCudaMemory(const void* ptr) {
#ifdef XCHAINER_ENABLE_CUDA
    cudaPointerAttributes attr = {};
    cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
    switch (status) {
        case cudaSuccess:
            return true;
        case cudaErrorInvalidValue:
            return false;
        default:
            cuda::CheckError(status);
            break;
    }
#else
    return false;
#endif  // XCHAINER_ENABLE_CUDA
}

std::shared_ptr<void> Allocate(size_t size) {
    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
        return std::make_unique<uint8_t[]>(size);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        void* raw_ptr = nullptr;
        cuda::CheckError(cudaMallocManaged(&raw_ptr, size, cudaMemAttachGlobal));
        return std::shared_ptr<void>{raw_ptr, cudaFree};
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t size) {
    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
#ifdef XCHAINER_ENABLE_CUDA
        if (IsPointerCudaMemory(src_ptr)) {
            cuda::CheckError(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
        } else {
#endif  // XCHAINER_ENABLE_CUDA
            std::memcpy(dst_ptr, src_ptr, size);
#ifdef XCHAINER_ENABLE_CUDA
        }
    } else if (device == MakeDevice("cuda")) {
        if (IsPointerCudaMemory(src_ptr)) {
            cuda::CheckError(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice));
        } else {
            cuda::CheckError(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyHostToDevice));
        }
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

std::shared_ptr<void> MemoryFromBuffer(const std::shared_ptr<void>& src_ptr, size_t size) {
#ifdef XCHAINER_ENABLE_CUDA
    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
        if (IsPointerCudaMemory(src_ptr.get())) {
            std::shared_ptr<void> dst_ptr = Allocate(size);
            cuda::CheckError(cudaMemcpy(dst_ptr.get(), src_ptr.get(), size, cudaMemcpyDeviceToHost));
            return dst_ptr;
        } else {
            return src_ptr;
        }
    } else if (device == MakeDevice("cuda")) {
        if (IsPointerCudaMemory(src_ptr.get())) {
            return src_ptr;
        } else {
            std::shared_ptr<void> dst_ptr = Allocate(size);
            cuda::CheckError(cudaMemcpy(dst_ptr.get(), src_ptr.get(), size, cudaMemcpyHostToDevice));
            return dst_ptr;
        }
    } else {
        throw DeviceError("invalid device");
    }
#else
    return src_ptr;
#endif  // XCHAINER_ENABLE_CUDA
}

}  // namespace
