#pragma once

#include <memory>
#include <string>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/scalar.h"

namespace xchainer {

class Backend {
public:
    Backend(std::string name) : name_(std::move(name)) {}
    virtual ~Backend() = default;

    virtual std::shared_ptr<void> Allocate(const Device& device, size_t bytesize) = 0;
    virtual void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) = 0;
    virtual std::shared_ptr<void> FromBuffer(const Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize) = 0;

    virtual void Fill(Array& out, Scalar value) = 0;

    virtual void Add(const Array& lhs, const Array& rhs, Array& out) = 0;
    virtual void Mul(const Array& lhs, const Array& rhs, Array& out) = 0;

    virtual void Synchronize() = 0;

    const std::string& name() const { return name_; }

private:
    std::string name_;
};

}  // namespace xchainer
