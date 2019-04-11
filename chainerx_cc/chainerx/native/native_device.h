#pragma once

#include <cstdint>
#include <memory>
#include <tuple>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/device.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/routines/pooling.h"
#include "chainerx/scalar.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace native {

class NativeDevice : public Device {
public:
    void Synchronize() override;

    // memory.cc

    std::shared_ptr<void> Allocate(size_t bytesize) override;

    void MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) override;

    void MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) override;

    std::shared_ptr<void> TransferDataFrom(
            Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) override;

    // reduction.cc

    void Sum(const Array& a, const Axes& axis, const Array& out) override;
    void AMax(const Array& a, const Axes& axis, const Array& out) override;

    // activation.cc

    void IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override;

    void IfGreaterElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override;
    void IfGreaterElseAAAA(const Array& x1, const Array& x2, const Array& pos, const Array& neg, const Array& out) override;

    void Tanh(const Array& x, const Array& out) override;

    // exp_log.cc

    void Exp(const Array& x, const Array& out) override;
    void Log(const Array& x, const Array& out) override;

    // misc.cc

    void Square(const Array& x, const Array& out) override;

    void Sqrt(const Array& x, const Array& out) override;

    void IsNan(const Array& x, const Array& out) override;
    void IsInf(const Array& x, const Array& out) override;

    // pool.cc

    std::unique_ptr<MaxPoolForwardBackward> GetMaxPoolForwardBackward(
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) override;

    std::unique_ptr<AveragePoolForwardBackward> GetAveragePoolForwardBackward(
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            AveragePoolPadMode pad_mode) override;

protected:
    NativeDevice(NativeBackend& backend, int index) : Device(backend, index) {}

private:
    friend NativeDevice* native_internal::CreateDevice(NativeBackend&, int);
};

}  // namespace native
}  // namespace chainerx
