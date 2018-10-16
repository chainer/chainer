#include "chainerx/cuda/cuda_device.h"

#include <cstddef>
#include <memory>

#include <cuda_runtime.h>

#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/device.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/native/native_device.h"

namespace chainerx {
namespace cuda {

std::shared_ptr<void> CudaDevice::Allocate(size_t bytesize) {
    void* ptr = memory_pool_.Malloc(bytesize);
    return std::shared_ptr<void>{ptr, [this](void* ptr) { memory_pool_.Free(ptr); }};
}

std::shared_ptr<void> CudaDevice::AllocatePinnedMemory(size_t bytesize) {
    void* ptr = pinned_memory_pool_.Malloc(bytesize);
    return std::shared_ptr<void>{ptr, [this](void* ptr) { pinned_memory_pool_.Free(ptr); }};
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
    // TODO(niboshi): Do device management with RAII
    int old_device{};
    CheckCudaError(cudaGetDevice(&old_device));
    CheckCudaError(cudaSetDevice(index()));
    if (&src_device == this || nullptr != dynamic_cast<CudaDevice*>(&src_device)) {
        // Copy between CUDA devices
        CheckCudaError(cudaMemcpyAsync(dst, src, bytesize, cudaMemcpyDeviceToDevice));
    } else {
        CHAINERX_ASSERT(
                nullptr != dynamic_cast<native::NativeDevice*>(&src_device) &&
                "CudaDevice only supports copy between cuda or native devices.");
        // Copy from native device
        std::shared_ptr<void> pinned_src_ptr = AllocatePinnedMemory(bytesize);
        CheckCudaError(cudaMemcpy(pinned_src_ptr.get(), src, bytesize, cudaMemcpyHostToHost));
        CheckCudaError(cudaMemcpyAsync(dst, pinned_src_ptr.get(), bytesize, cudaMemcpyHostToDevice));
    }
    CheckCudaError(cudaSetDevice(old_device));
}

void CudaDevice::MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) {
    std::cout << "memory copy to 1" << std::endl;
    CHAINERX_ASSERT(bytesize == 0 || src == nullptr || IsPointerCudaMemory(src));
    if (bytesize == 0) {
        return;
    }
    std::cout << "memory copy to 2" << std::endl;
    // TODO(niboshi): Do device management with RAII
    int old_device{};
    std::cout << "memory copy to 3" << std::endl;
    CheckCudaError(cudaGetDevice(&old_device));
    std::cout << "memory copy to 4" << std::endl;
    CheckCudaError(cudaSetDevice(index()));
    std::cout << "memory copy to 5" << std::endl;
    if (&dst_device == this || nullptr != dynamic_cast<CudaDevice*>(&dst_device)) {
        // Copy between CUDA devices
        std::cout << "memory copy to 6" << std::endl;
        CheckCudaError(cudaMemcpyAsync(dst, src, bytesize, cudaMemcpyDeviceToDevice));
        std::cout << "memory copy to 7" << std::endl;
    } else {
        std::cout << "memory copy to 8" << std::endl;
        CHAINERX_ASSERT(
                nullptr != dynamic_cast<native::NativeDevice*>(&dst_device) &&
                "CudaDevice only supports copy between cuda or native devices.");
        // Copy to native device
        std::cout << "memory copy to 9" << std::endl;
        CheckCudaError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToHost));
        std::cout << "memory copy to 10" << std::endl;
    }
    std::cout << "memory copy to 11" << std::endl;
    CheckCudaError(cudaSetDevice(old_device));
    std::cout << "memory copy to 12" << std::endl;
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
    std::cout << "from host 1" << std::endl;
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    std::cout << "from host 2" << std::endl;
    std::shared_ptr<void> pinned_src_ptr = AllocatePinnedMemory(bytesize);
    std::cout << "from host 3" << std::endl;
    int old_device{};
    CheckCudaError(cudaGetDevice(&old_device));
    CheckCudaError(cudaSetDevice(index()));
    CheckCudaError(cudaMemcpy(pinned_src_ptr.get(), src_ptr.get(), bytesize, cudaMemcpyHostToHost));
    std::cout << "from host 4" << std::endl;
    CheckCudaError(cudaMemcpyAsync(dst_ptr.get(), pinned_src_ptr.get(), bytesize, cudaMemcpyHostToDevice));
    CheckCudaError(cudaSetDevice(old_device));
    std::cout << "from host 5" << std::endl;
    return dst_ptr;
}

}  // namespace cuda
}  // namespace chainerx
