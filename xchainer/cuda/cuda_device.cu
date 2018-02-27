#include "xchainer/cuda/cuda_device.h"

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/dtype.h"
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

}  // namespace

std::shared_ptr<void> CudaDevice::Allocate(size_t bytesize) {
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

void CudaDevice::MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) {
    (void)dst_ptr;   // unused
    (void)src_ptr;   // unused
    (void)bytesize;  // unused
}

std::shared_ptr<void> CudaDevice::FromBuffer(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    (void)src_ptr;   // unused
    (void)bytesize;  // unused
    return nullptr;
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

void CudaDevice::Add(const Array& lhs, const Array& rhs, Array& out) {
    (void)lhs;  // unused
    (void)rhs;  // unused
    (void)out;  // unused
}

void CudaDevice::Mul(const Array& lhs, const Array& rhs, Array& out) {
    (void)lhs;  // unused
    (void)rhs;  // unused
    (void)out;  // unused
}

void CudaDevice::Synchronize() {
    CheckError(cudaSetDevice(index()));
    CheckError(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace xchainer
