#pragma once

#include "xchainer/backend.h"

namespace xchainer {
namespace cuda {

class CudaBackend : public Backend {
public:
    std::shared_ptr<void> Allocate(const Device& device, size_t bytesize) const;
    void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) const;
    std::shared_ptr<void> FromBuffer(const Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize) const;

    void Fill(Array& out, Scalar value) const;

    void Add(const Array& lhs, const Array& rhs, Array& out) const;
    void Mul(const Array& lhs, const Array& rhs, Array& out) const;

    void Synchronize() const;
};

}  // namespace cuda
}  // namespace xchainer
