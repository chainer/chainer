#pragma once

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/device.h"

namespace xchainer {
namespace cuda {

class CudaDevice : public Device {
public:
    CudaDevice(CudaBackend& backend, int index) : Device(backend, index) {}

    std::shared_ptr<void> Allocate(size_t bytesize) override;
    void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) override;
    std::shared_ptr<void> FromBuffer(const std::shared_ptr<void>& src_ptr, size_t bytesize) override;

    void Fill(Array& out, Scalar value) override;

    void Add(const Array& lhs, const Array& rhs, Array& out) override;
    void Mul(const Array& lhs, const Array& rhs, Array& out) override;

    void Synchronize() override;
};

}  // namespace cuda
}  // namespace xchainer
