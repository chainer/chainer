#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

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

// Creates an array with given contiguous data
Array FromContiguousHostData(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, Device& device = GetDefaultDevice());

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
Array Arange(Scalar start, Scalar stop, Dtype dtype, Device& device = GetDefaultDevice());
Array Arange(Scalar start, Scalar stop, Device& device = GetDefaultDevice());
Array Arange(Scalar stop, Dtype dtype, Device& device = GetDefaultDevice());
Array Arange(Scalar stop, Device& device = GetDefaultDevice());

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

// Creates the identity array.
Array Identity(int64_t n, Dtype dtype, Device& device = GetDefaultDevice());

// Creates a 2-dimensional array with ones along the k-th diagonal and zeros elsewhere.
Array Eye(int64_t N, int64_t M, int64_t k, Dtype dtype, Device& device = GetDefaultDevice());

}  // namespace xchainer
