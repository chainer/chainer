#pragma once

#include "xchainer/backend.h"

namespace xchainer {
namespace cuda {

class CudaBackend : public Backend {
public:
    void Fill(Array& out, Scalar value);
    void Add(const Array& lhs, const Array& rhs, Array& out);
    void Mul(const Array& lhs, const Array& rhs, Array& out);
    std::shared_ptr<void> Allocate(const Device& device, size_t bytesize);
    void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize);
    std::shared_ptr<void> FromBuffer(const Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize);
    void Synchronize();
};

}  // namespace cuda
}  // namespace xchainer
