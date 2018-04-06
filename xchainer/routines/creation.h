#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace internal {

// Returns the minimum number of bytes required to pack the data with specified strides and shape.
size_t GetRequiredBytes(const Shape& shape, const Strides& strides, size_t element_size);

// Creates an array with given data packed with specified strides
Array FromHostData(
        const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, const Strides& strides, Device& device = GetDefaultDevice());

// Creates an empty array with specified strides.
Array Empty(const Shape& shape, Dtype dtype, const Strides& strides, Device& device = GetDefaultDevice());

}  // namespace internal

Array Empty(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());
Array Full(const Shape& shape, Scalar fill_value, Dtype dtype, Device& device = GetDefaultDevice());
Array Full(const Shape& shape, Scalar fill_value, Device& device = GetDefaultDevice());
Array Zeros(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());
Array Ones(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());
Array Arange(Scalar start, Scalar stop, Scalar step, Dtype dtype, Device& device = GetDefaultDevice());
Array Arange(Scalar start, Scalar stop, Scalar step, Device& device = GetDefaultDevice());

// Creates an array which has the same shape and dtype as the other array.
// The new array is allocated in the default device. The device of the other array
// is ignored.
Array EmptyLike(const Array& a, Device& device = GetDefaultDevice());
Array FullLike(const Array& a, Scalar fill_value, Device& device = GetDefaultDevice());
Array ZerosLike(const Array& a, Device& device = GetDefaultDevice());
Array OnesLike(const Array& a, Device& device = GetDefaultDevice());

// Creates a copy.
// It will be connected to all the graphs.
// It will be always C-contiguous.
Array Copy(const Array& a);

}  // namespace xchainer
