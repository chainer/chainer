#include "chainerx/cuda/cuda_device.h"

#include <cstddef>
#include <memory>

#include <cuda_runtime.h>

#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/memory_pool.h"
#include "chainerx/device.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/native/native_device.h"

namespace chainerx {
namespace cuda {

std::shared_ptr<void> CudaDevice::Allocate(size_t bytesize) {
    auto deleter = [weak_pool = std::weak_ptr<MemoryPool>{device_memory_pool_}](void* ptr) {
        if (std::shared_ptr<MemoryPool> pool = weak_pool.lock()) {
            pool->FreeNoExcept(ptr);
        }
    };
    return std::shared_ptr<void>{device_memory_pool_->Malloc(bytesize), std::move(deleter)};
}

std::shared_ptr<void> CudaDevice::AllocatePinnedMemory(size_t bytesize) {
    auto deleter = [weak_pool = std::weak_ptr<MemoryPool>{pinned_memory_pool_}](void* ptr) {
        if (std::shared_ptr<MemoryPool> pool = weak_pool.lock()) {
            pool->FreeNoExcept(ptr);
        }
    };
    return std::shared_ptr<void>{pinned_memory_pool_->Malloc(bytesize), std::move(deleter)};
}

void CudaDevice::MemoryCopyFromHostAsync(void* dst, const void* src, size_t bytesize) {
    CudaSetDeviceScope scope{index()};

    memory_keeper_.Collect();

    // Allocate a pinned memory and copy the host data.
    std::shared_ptr<void> pinned_src_ptr = AllocatePinnedMemory(bytesize);
    std::memcpy(pinned_src_ptr.get(), src, bytesize);

    // Transfer from the pinned memory to the device.
    CheckCudaError(cudaMemcpyAsync(dst, pinned_src_ptr.get(), bytesize, cudaMemcpyHostToDevice));

    // Register the pinned memory to the keeper to prevent it from being freed before host-to-device transfer is finished.
    memory_keeper_.Add(nullptr, std::move(pinned_src_ptr));
}

std::shared_ptr<void> CudaDevice::MakeDataFromForeignPointer(const std::shared_ptr<void>& data) {
    if (data == nullptr) {
        return data;
    }

    // check memory validity
    void* ptr = data.get();
    cudaPointerAttributes attr{};
    cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
    switch (status) {
        case cudaSuccess:
            if (attr.device != index()) {
                throw ChainerxError{"CUDA memory: ", ptr, " must reside on the device: ", index()};
            }
            break;
        case cudaErrorInvalidValue:
            throw ChainerxError{"Memory: ", ptr, " is not a CUDA memory"};
        default:
            Throw(status);
    }
    return data;
}

void CudaDevice::MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) {
    CHAINERX_ASSERT(bytesize == 0 || IsPointerCudaMemory(dst));
    if (bytesize == 0) {
        return;
    }
    CudaSetDeviceScope scope{index()};
    if (&src_device == this || nullptr != dynamic_cast<CudaDevice*>(&src_device)) {
        // Copy between CUDA devices
        CheckCudaError(cudaMemcpyAsync(dst, src, bytesize, cudaMemcpyDeviceToDevice));
    } else {
        CHAINERX_ASSERT(
                nullptr != dynamic_cast<native::NativeDevice*>(&src_device) &&
                "CudaDevice only supports copy between cuda or native devices.");
        // Copy from native device
        MemoryCopyFromHostAsync(dst, src, bytesize);
    }
}

void CudaDevice::MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) {
    CHAINERX_ASSERT(bytesize == 0 || src == nullptr || IsPointerCudaMemory(src));
    if (bytesize == 0) {
        return;
    }
    CudaSetDeviceScope scope{index()};
    if (&dst_device == this || nullptr != dynamic_cast<CudaDevice*>(&dst_device)) {
        // Copy between CUDA devices
        CheckCudaError(cudaMemcpyAsync(dst, src, bytesize, cudaMemcpyDeviceToDevice));
    } else {
        CHAINERX_ASSERT(
                nullptr != dynamic_cast<native::NativeDevice*>(&dst_device) &&
                "CudaDevice only supports copy between cuda or native devices.");
        // Copy to native device
        CheckCudaError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToHost));
    }
}

std::shared_ptr<void> CudaDevice::TransferDataFrom(
        Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    MemoryCopyFrom(dst_ptr.get(), &(static_cast<int8_t*>(src_ptr.get())[offset]), bytesize, src_device);
    return dst_ptr;
}

std::shared_ptr<void> CudaDevice::TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = dst_device.Allocate(bytesize);
    MemoryCopyTo(dst_ptr.get(), &(static_cast<int8_t*>(src_ptr.get())[offset]), bytesize, dst_device);
    return dst_ptr;
}

std::shared_ptr<void> CudaDevice::FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    MemoryCopyFromHostAsync(dst_ptr.get(), src_ptr.get(), bytesize);
    return dst_ptr;
}

}  // namespace cuda
}  // namespace chainerx
