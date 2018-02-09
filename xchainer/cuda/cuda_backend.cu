#include "xchainer/cuda/cuda_backend.h"

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

std::shared_ptr<void> CudaBackend::Allocate(const Device& device, size_t bytesize) {
    (void)device;    // unused
    (void)bytesize;  // unused
    return nullptr;
}

void CudaBackend::MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) {
    (void)dst_ptr;   // unused
    (void)src_ptr;   // unused
    (void)bytesize;  // unused
}

std::shared_ptr<void> CudaBackend::FromBuffer(const Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    (void)device;    // unused
    (void)src_ptr;   // unused
    (void)bytesize;  // unused
    return nullptr;
}

void CudaBackend::Fill(Array& out, Scalar value) {
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

void CudaBackend::Add(const Array& lhs, const Array& rhs, Array& out) {
    (void)lhs;  // unused
    (void)rhs;  // unused
    (void)out;  // unused
}

void CudaBackend::Mul(const Array& lhs, const Array& rhs, Array& out) {
    (void)lhs;  // unused
    (void)rhs;  // unused
    (void)out;  // unused
}

void CudaBackend::Synchronize() { CheckError(cudaDeviceSynchronize()); }

}  // namespace cuda
}  // namespace xchainer
