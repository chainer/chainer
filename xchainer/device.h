#pragma once

#include <memory>
#include <string>

#include "xchainer/array.h"
#include "xchainer/backend.h"
#include "xchainer/scalar.h"

namespace xchainer {

// Device base class.
// Note that these member functions may be called from the framework or user code.
class Device {
public:
    Device(Backend& backend, int index) : backend_(backend), index_(index) {}
    virtual ~Device() = default;

    // Allocates a memory chunk on this device.
    virtual std::shared_ptr<void> Allocate(size_t bytesize) = 0;

    // Copies the data between two memory chunks.
    // The caller must guarantee that:
    // - both dst_ptr and src_ptr reside in this device.
    // - a copy between these memory regions can be done transparently, e.g. without sychronization,
    // - and these memory regions are not overlapped.
    virtual void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) = 0;

    // Creates a data buffer filled with the specified data on this device.
    //
    // It may allocate a new memory or return an alias.
    // src_ptr must reside in the main RAM.
    virtual std::shared_ptr<void> FromBuffer(const std::shared_ptr<void>& src_ptr, size_t bytesize) = 0;

    virtual void Fill(Array& out, Scalar value) = 0;

    virtual void Add(const Array& lhs, const Array& rhs, Array& out) = 0;
    virtual void Mul(const Array& lhs, const Array& rhs, Array& out) = 0;

    virtual void Synchronize() = 0;

    // TODO(sonots): optimize string concat
    std::string name() const { return backend_.name() + ":" + std::to_string(index_); }

    Backend& backend() const { return backend_; }
    int index() const { return index_; }

private:
    Backend& backend_;
    int index_;
};

}  // namespace xchainer
