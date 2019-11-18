#pragma once

#include <cstdint>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/shape.h"

namespace chainerx {

// Retrieves a scalar from a single-element array.
//
// If the array is not single-element, DimensionError is thrown.
Scalar AsScalar(const Array& a);

Array Ravel(const Array& a);

// Returns a view where the specified axis is moved to start.
Array RollAxis(const Array& a, int8_t axis, int8_t start = 0);

// Returns a transposed view of the array.
Array Transpose(const Array& a, const OptionalAxes& axes = absl::nullopt);

// Returns a reshaped array.
Array Reshape(const Array& a, const Shape& newshape);

// Returns a squeezed array with unit-length axes removed.
//
// If no axes are specified, all axes of unit-lengths are removed.
// If no axes can be removed, an array with aliased data is returned.
Array Squeeze(const Array& a, const OptionalAxes& axis = absl::nullopt);

// Broadcasts the array to the specified shape.
// Returned array is always a view to this array.
Array BroadcastTo(const Array& array, const Shape& shape);

// Returns a concatenated array.
Array Concatenate(const std::vector<Array>& arrays);
Array Concatenate(const std::vector<Array>& arrays, absl::optional<int8_t> axis);

// Returns a joined array along a new axis.
Array Stack(const std::vector<Array>& arrays, int8_t axis = 0);

// Returns a set of arrays resulting from splitting the given array into sections along the specified axis.
// If the dimension is not equally divisible, DimensionError is throws.
std::vector<Array> Split(const Array& ary, int64_t sections, int8_t axis = 0);

// Returns a set of arrays resulting from splitting the given array at the indices along the specified axis.
std::vector<Array> Split(const Array& ary, std::vector<int64_t> indices, int8_t axis = 0);

std::vector<Array> DSplit(const Array& ary, int64_t sections);

std::vector<Array> DSplit(const Array& ary, std::vector<int64_t> indices);

std::vector<Array> VSplit(const Array& ary, int64_t sections);

std::vector<Array> VSplit(const Array& ary, std::vector<int64_t> indices);

std::vector<Array> HSplit(const Array& ary, int64_t sections);

std::vector<Array> HSplit(const Array& ary, std::vector<int64_t> indices);

Array Swapaxes(const Array& a, int8_t axis1, int8_t axis2);

Array Repeat(const Array& a, int64_t repeats, absl::optional<int8_t> axis);

Array Repeat(const Array& a, const std::vector<int64_t>& repeats, absl::optional<int8_t> axis);

Array ExpandDims(const Array& a, int8_t axis);

Array Flip(const Array& m, const OptionalAxes& axes = absl::nullopt);

Array Fliplr(const Array& m);

Array Flipud(const Array& m);

Array AtLeast2D(const Array& x);

Array AtLeast3D(const Array& x);

// Returns a joined array along horizontal axis.
Array HStack(const std::vector<Array>& arrays);

// Returns a joined array along vertical axis.
Array VStack(const std::vector<Array>& arrays);

Array DStack(const std::vector<Array>& arrays);

Array Moveaxis(const Array& a, const Axes& source, const Axes& destination);

enum class CastingMode {
    kNo,
};

void CopyTo(const Array& dst, const Array& src, CastingMode casting, const Array& where);

}  // namespace chainerx
