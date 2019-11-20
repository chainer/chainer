#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/axes.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace internal {

// Returns the minimum number of bytes required to pack the data with specified strides and shape.
// TODO(niboshi): Replace this with chainerx::GetDataRange()
size_t GetRequiredBytes(const Shape& shape, const Strides& strides, size_t item_size);

// Creates an array with given data packed with specified strides
Array FromHostData(
        const Shape& shape,
        Dtype dtype,
        const std::shared_ptr<void>& data,
        const Strides& strides,
        int64_t offset,
        Device& device = GetDefaultDevice());

// Creates an empty array with specified strides.
Array Empty(const Shape& shape, Dtype dtype, const Strides& strides, Device& device = GetDefaultDevice());

// Creates an empty array with reduced shape.
Array EmptyReduced(const Shape& shape, Dtype dtype, const Axes& axes, bool keepdims, Device& device = GetDefaultDevice());

}  // namespace internal

// Creates an array with given contiguous data
Array FromContiguousHostData(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, Device& device = GetDefaultDevice());

// Creates an array with given data without copying.
//
// The data must reside in the specified device.
// If strides are not given, the data is considered as a contiguous data.
Array FromData(
        const Shape& shape,
        Dtype dtype,
        const std::shared_ptr<void>& data,
        const absl::optional<Strides>& strides = absl::nullopt,
        int64_t offset = 0,
        Device& device = GetDefaultDevice());

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
Array Eye(
        int64_t n, absl::optional<int64_t> m, absl::optional<int64_t> k, absl::optional<Dtype> dtype, Device& device = GetDefaultDevice());

// Returns a C-contiguous array without changing input shape.
Array AsContiguous(const Array& a, Dtype dtype);

// Returns a C-contiguous array with the same shape and dtype as the input array.
inline Array AsContiguous(const Array& a) { return AsContiguous(a, a.dtype()); }

// Returns a C-contiguous array.
// An input array with shape {} results in a new array with shape {1}.
Array AsContiguousArray(const Array& a, absl::optional<Dtype> dtype = absl::nullopt);

Array Diag(const Array& v, int64_t k = 0);

Array Diagflat(const Array& v, int64_t k = 0);

// Creates a 1-d array with evenly spaced numbers.
Array Linspace(
        Scalar start,
        Scalar stop,
        absl::optional<int64_t> num = absl::nullopt,
        bool endpoint = true,
        absl::optional<Dtype> dtype = absl::nullopt,
        Device& device = GetDefaultDevice());

enum class MeshgridIndexingMode { kCartesian, kMatrix };

std::vector<Array> Meshgrid(const std::vector<Array>& arrays, MeshgridIndexingMode mode);

// Creates a 2-dimensional array with ones at and below the given diagonal and zeros elsewhere.
Array Tri(
        int64_t n, absl::optional<int64_t> m, absl::optional<int64_t> k, absl::optional<Dtype> dtype, Device& device = GetDefaultDevice());

// Creates a lower triangle of an array.
Array Tril(const Array& m, int64_t k);

// Creates an upper triangle of an array.
Array Triu(const Array& m, int64_t k);

}  // namespace chainerx
