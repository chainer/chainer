#include "xchainer/cuda/cuda_device.h"

#include <algorithm>
#include <memory>
#include <tuple>
#include <utility>

#include <cuda_runtime.h>

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/memory.h"
#include "xchainer/native_device.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace cuda {

namespace {

template <typename T>
__global__ void FillKernel(T* odata, T value, int64_t total_size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        odata[i] = value;
    }
}

template <typename T>
__global__ void AddKernel(const T* ldata, const T* rdata, T* odata, int64_t total_size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        odata[i] = ldata[i] + rdata[i];
    }
}

template <typename T>
__global__ void MulKernel(const T* ldata, const T* rdata, T* odata, int64_t total_size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        odata[i] = ldata[i] * rdata[i];
    }
}

}  // namespace

std::shared_ptr<void> CudaDevice::Allocate(size_t bytesize) {
    if (bytesize == 0) {
        return nullptr;
    }
    CheckError(cudaSetDevice(index()));
    void* raw_ptr = nullptr;
    // Be careful to be exception-safe, i.e.,
    // do not throw any exceptions before creating shared_ptr when memory allocation is succeeded
    cudaError_t status = cudaMallocManaged(&raw_ptr, bytesize, cudaMemAttachGlobal);
    if (status != cudaSuccess) {
        cuda::Throw(status);
    }
    return std::shared_ptr<void>{raw_ptr, cudaFree};
}

void CudaDevice::MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) {
    assert(internal::IsPointerCudaMemory(dst));
    if (&src_device == this || nullptr != dynamic_cast<CudaDevice*>(&src_device)) {
        // Copy between CUDA devices
        CheckError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToDevice));
    } else {
        assert(nullptr != dynamic_cast<NativeDevice*>(&src_device) && "CudaDevice only supports copy between cuda or native devices.");
        // Copy from native device
        CheckError(cudaMemcpy(dst, src, bytesize, cudaMemcpyHostToDevice));
    }
}

void CudaDevice::MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) {
    assert(internal::IsPointerCudaMemory(src));
    if (&dst_device == this || nullptr != dynamic_cast<CudaDevice*>(&dst_device)) {
        // Copy between CUDA devices
        CheckError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToDevice));
    } else {
        assert(nullptr != dynamic_cast<NativeDevice*>(&dst_device) && "CudaDevice only supports copy between cuda or native devices.");
        // Copy to native device
        CheckError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToHost));
    }
}

std::tuple<std::shared_ptr<void>, size_t> CudaDevice::TransferDataFrom(Device& src_device, const std::shared_ptr<void>& src_ptr,
                                                                       size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    MemoryCopyFrom(dst_ptr.get(), &static_cast<int8_t*>(src_ptr.get())[offset], bytesize, src_device);
    return std::make_tuple(std::move(dst_ptr), 0);
}

std::tuple<std::shared_ptr<void>, size_t> CudaDevice::TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr,
                                                                     size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = dst_device.Allocate(bytesize);
    MemoryCopyTo(dst_ptr.get(), &static_cast<int8_t*>(src_ptr.get())[offset], bytesize, dst_device);
    return std::make_tuple(std::move(dst_ptr), 0);
}

std::shared_ptr<void> CudaDevice::FromBuffer(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    CheckError(cudaMemcpy(dst_ptr.get(), src_ptr.get(), bytesize, cudaMemcpyHostToDevice));
    return dst_ptr;
}

void CudaDevice::Fill(Array& out, Scalar value) {
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&FillKernel<T>).block_size;

        int64_t total_size = out.GetTotalSize();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        auto* odata = static_cast<T*>(out.data().get());
        FillKernel<<<grid_size, block_size>>>(odata, static_cast<T>(value), total_size);
    });
}

// TODO(sonots): support stream
void CudaDevice::Add(const Array& lhs, const Array& rhs, Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    cudaSetDevice(index());
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&AddKernel<T>).block_size;

        int64_t total_size = lhs.GetTotalSize();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());
        AddKernel<<<grid_size, block_size>>>(ldata, rdata, odata, total_size);
    });
}

// TODO(sonots): support stream
void CudaDevice::Mul(const Array& lhs, const Array& rhs, Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    cudaSetDevice(index());
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&MulKernel<T>).block_size;

        int64_t total_size = lhs.GetTotalSize();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());
        MulKernel<<<grid_size, block_size>>>(ldata, rdata, odata, total_size);
    });
}

void CudaDevice::Synchronize() {
    CheckError(cudaSetDevice(index()));
    CheckError(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace xchainer
