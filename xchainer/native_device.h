#pragma once

#include <memory>
#include <tuple>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/native_backend.h"

namespace xchainer {

class NativeDevice : public Device {
public:
    NativeDevice(NativeBackend& backend, int index) : Device(backend, index) {}

    std::shared_ptr<void> Allocate(size_t bytesize) override;

    void MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) override;

    void MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) override;

    std::tuple<std::shared_ptr<void>, size_t> TransferDataFrom(Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset,
                                                               size_t bytesize) override;

    std::tuple<std::shared_ptr<void>, size_t> TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset,
                                                             size_t bytesize) override;

    std::shared_ptr<void> FromBuffer(const std::shared_ptr<void>& src_ptr, size_t bytesize) override;

    void Fill(Array& out, Scalar value) override;

    void Add(const Array& lhs, const Array& rhs, Array& out) override;
    void Mul(const Array& lhs, const Array& rhs, Array& out) override;

    void Synchronize() override;
};

}  // namespace xchainer
