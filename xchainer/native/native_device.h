#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/native_backend.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace native {

class NativeDevice : public Device {
public:
    NativeDevice(NativeBackend& backend, int index) : Device(backend, index) {}

    std::shared_ptr<void> Allocate(size_t bytesize) override;

    void MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) override;

    void MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) override;

    std::shared_ptr<void> TransferDataFrom(
            Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) override;

    void Fill(const Array& out, Scalar value) override;

    void Arange(Scalar start, Scalar step, const Array& out) override;

    void ArgMax(const Array& a, const std::vector<int8_t>& axis, const Array& out) override;

    void Sum(const Array& a, const std::vector<int8_t>& axis, const Array& out) override;

    void Copy(const Array& a, const Array& out) override;

    void Equal(const Array& x1, const Array& x2, const Array& out) override;

    void Add(const Array& x1, const Array& x2, const Array& out) override;
    void Subtract(const Array& x1, const Array& x2, const Array& out) override;
    void Multiply(const Array& x1, const Array& x2, const Array& out) override;
    void MultiplyAS(const Array& x1, Scalar x2, const Array& out) override;
    void Divide(const Array& lhs, const Array& rhs, const Array& out) override;

    void IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override;

    void Dot(const Array& a, const Array& b, const Array& out) override;

    void Exp(const Array& x, const Array& out) override;
    void Log(const Array& x, const Array& out) override;

    void Synchronize() override;
};

}  // namespace native
}  // namespace xchainer
