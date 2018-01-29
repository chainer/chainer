#pragma once

#include "xchainer/backend.h"

namespace xchainer {

class NativeBackend : public Backend {
public:
    std::shared_ptr<void> Allocate(const Device& device, size_t bytesize);
    void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize);
    std::shared_ptr<void> FromBuffer(const Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize);

    void Fill(Array& out, Scalar value);

    void Add(const Array& lhs, const Array& rhs, Array& out);
    void Mul(const Array& lhs, const Array& rhs, Array& out);

    void Synchronize();
};

}  // namespace xchainer
