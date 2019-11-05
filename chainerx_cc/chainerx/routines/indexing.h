#pragma once

#include <cstdint>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace internal {

// Returns a view selected with the indices.
Array At(const Array& a, const std::vector<ArrayIndex>& indices);

}  // namespace internal

// Adds each slice of `b` along the axis `axis` to `a`'s corresponding slices, specified by `indices`.
// The resulting array is returned. It is not in-place operation: the input arrays are not altered.
//
// `axis` must be within [0, b.ndim()).
// `indices` must have dtype kind of either kInt or kUInt.
//
// It is differentiable with respect to `a` and `b`.
Array AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, IndexBoundsMode mode = IndexBoundsMode::kDefault);

// Takes elements specified by indices from an array.
// Indices that are out of bounds are wrapped around.
//
// `axis` must be within [0, a.ndim()).
// `indices` must have dtype kind of either kInt or kUInt.
//
// TODO(niboshi): Support Scalar and StackVector as indices.
// TODO(niboshi): Support axis=None behavior in NumPy.
Array Take(const Array& a, const Array& indices, int8_t axis, IndexBoundsMode mode = IndexBoundsMode::kDefault);

Array Where(const Array& condition, const Array& x, const Array& y);

Array Where(const Array& condition, const Array& x, Scalar y);

Array Where(const Array& condition, Scalar x, const Array& y);

Array Where(const Array& condition, Scalar x, Scalar y);

std::vector<Array> Nonzero(const Array& a);

}  // namespace chainerx
