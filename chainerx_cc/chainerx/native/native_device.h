#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/kernels/pooling.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/routines/pooling.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace native {

class NativeMaxPoolGradState : public MaxPoolGradState {
public:
    NativeMaxPoolGradState(Array x, Array col, Axes axes) : x_{std::move(x)}, col_{std::move(col)}, axes_{std::move(axes)} {}

    const Array& x() const { return x_; }
    const Array& col() const { return col_; }
    const Axes& axes() const { return axes_; }

private:
    Array x_{};
    Array col_{};
    Axes axes_{};
};

class NativeMaxPoolGradGradState : public MaxPoolGradGradState {
public:
    NativeMaxPoolGradGradState(Array indices, Array offset, Dtype x_dtype)
        : indices_{std::move(indices)}, offset_{std::move(offset)}, x_dtype_{x_dtype} {}

    const Array& indices() const { return indices_; }
    const Array& offset() const { return offset_; }
    Dtype x_dtype() const { return x_dtype_; }

private:
    Array indices_{};
    Array offset_{};
    Dtype x_dtype_{};
};

class NativeAveragePoolGradState : public AveragePoolGradState {
public:
    NativeAveragePoolGradState(Array x, Shape gcol_shape, absl::optional<Array> width_ignore)
        : x_{std::move(x)}, gcol_shape_{std::move(gcol_shape)}, width_ignore_{std::move(width_ignore)} {}

    const Array& x() const { return x_; }
    const Shape& gcol_shape() const { return gcol_shape_; }
    const absl::optional<Array>& width_ignore() const { return width_ignore_; }

private:
    Array x_;
    Shape gcol_shape_;
    absl::optional<Array> width_ignore_;
};

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

protected:
    NativeDevice(NativeBackend& backend, int index) : Device(backend, index) {}

private:
    friend NativeDevice* native_internal::CreateDevice(NativeBackend& backend, int index);
};

}  // namespace native
}  // namespace chainerx
