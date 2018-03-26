#include <cmath>
#include <cstddef>
#include <memory>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace routines {
namespace internal {

size_t GetRequiredBytes(const Shape& shape, const Strides& strides, size_t element_size) {
    assert(shape.ndim() == strides.ndim());

    // Calculate the distance between the first and the last element, plus single element size.
    size_t total_bytes = element_size;
    for (int8_t i = 0; i < shape.ndim(); ++i) {
        total_bytes += (shape[i] - 1) * std::abs(strides[i]);
    }
    return total_bytes;
}

Array ArrayFromBuffer(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, const Strides& strides, Device& device) {
    auto bytesize = GetRequiredBytes(shape, strides, GetElementSize(dtype));
    std::shared_ptr<void> device_data = device.FromBuffer(data, bytesize);
    return {shape, strides, dtype, device, device_data};
}

}  // namespace internal

Array FromBuffer(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, Device& device) {
    return internal::ArrayFromBuffer(shape, dtype, data, {shape, dtype}, device);
}

}  // namespace routines
}  // namespace xchainer
