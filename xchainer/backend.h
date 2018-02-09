#pragma once

#include <memory>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/scalar.h"

namespace xchainer {


// Backend base class.
// Note that these member functions may be called from the framework or user code.
class Backend {
public:
    virtual ~Backend() = default;

    // Allocates a memory chunk on the specified device.
    virtual std::shared_ptr<void> Allocate(const Device& device, size_t bytesize) = 0;

    // Copies the data between two memory chunks.
    // The caller must guarantee that:
    // - both dst_ptr and str_ptr reside in devices of this backend,
    // - a copy between these memory regions can be done transparently,
    // - and these memory regions are not overlapped.
    virtual void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) = 0;

    // Creates a data buffer filled with the specified data on the specified device.
    //
    // It may allocate a new memory or return an alias.
    // src_ptr is guaranteed to reside in the main RAM.
    virtual std::shared_ptr<void> FromBuffer(const Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize) = 0;

    virtual void Fill(Array& out, Scalar value) = 0;

    virtual void Add(const Array& lhs, const Array& rhs, Array& out) = 0;
    virtual void Mul(const Array& lhs, const Array& rhs, Array& out) = 0;

    virtual void Synchronize() = 0;
};

}  // namespace xchainer
