#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_runtime.h"

namespace xchainer {
namespace cuda {

void CudaBackend::Fill(Array& out, Scalar value) {
    (void)out;    // unused
    (void)value;  // unused
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

void CudaBackend::Synchronize() { CheckError(cudaDeviceSynchronize()); }

}  // namespace cuda
}  // namespace xchainer
