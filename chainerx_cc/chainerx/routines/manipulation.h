#pragma once

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/shape.h"

namespace chainerx {

// Retrieves a scalar from a single-element array.
//
// If the array is not single-element, DimensionError is thrown.
Scalar AsScalar(const Array& a);

// Returns a view where the specified axis is moved to start.
Array RollAxis(const Array& a, int8_t axis, int8_t start = 0);

// Returns a transposed view of the array.
Array Transpose(const Array& a, const OptionalAxes& axes = nonstd::nullopt);

// Returns a reshaped array.
Array Reshape(const Array& a, const Shape& newshape);

// Returns a squeezed array with unit-length axes removed.
//
// If no axes are specified, all axes of unit-lengths are removed.
// If no axes can be removed, an array with aliased data is returned.
Array Squeeze(const Array& a, const OptionalAxes& axis = nonstd::nullopt);

// Broadcasts the array to the specified shape.
// Returned array is always a view to this array.
Array BroadcastTo(const Array& array, const Shape& shape);

// Returns a concatenated array.
Array Concatenate(const std::vector<Array>& arrays);
Array Concatenate(const std::vector<Array>& arrays, nonstd::optional<int8_t> axis);

}  // namespace chainerx
