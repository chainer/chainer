#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native_backend.h"

namespace xchainer {
namespace internal {

// Prepares indexable arrays and indexers for array reduction.
//
// It returns a tuple containing the following items in this order.
// - Source indexable array
// - Output indexable array
// - Source array indexer
// - Output array indexer
// - Reduction indexer
//
// Axes of the source indexable array are reordered so that output axes come first and reduction axes follow.
//
// In both source and output indexable arrays, 1-dim axes are eliminated.
template <typename T>
std::tuple<IndexableArray<const T>, IndexableArray<T>, Indexer<>, Indexer<>, Indexer<>> PrepareIndexableArraysForReduction(
        const Array& src, const std::vector<int8_t>& axis, Array& out);

}  // namespace internal

class NativeDevice : public Device {
public:
    NativeDevice(NativeBackend& backend, int index) : Device(backend, index) {}

    std::shared_ptr<void> Allocate(size_t bytesize) override;

    void MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) override;

    void MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) override;

    std::shared_ptr<void> TransferDataFrom(
            Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> FromBuffer(const std::shared_ptr<void>& src_ptr, size_t bytesize) override;

    void Fill(Array& out, Scalar value) override;

    void Sum(const Array& src, const std::vector<int8_t>& axis, Array& out) override;

    void Copy(const Array& src, Array& out) override;

    void Add(const Array& lhs, const Array& rhs, Array& out) override;
    void Mul(const Array& lhs, const Array& rhs, Array& out) override;

    void Synchronize() override;
};

}  // namespace xchainer
