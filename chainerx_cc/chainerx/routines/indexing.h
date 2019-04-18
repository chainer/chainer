#pragma once

#include <cstdint>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/array_index.h"

namespace chainerx {
namespace internal {

// Returns a view selected with the indices.
Array At(const Array& a, const std::vector<ArrayIndex>& indices);

}  // namespace internal

// Adds each slice of `b` along the axis `axis` to `a`'s corresponding slices, specified by `indices`.
// Input arrays `a`, `indices`, and `b` are not altered.
//
// TODO(niboshi): This function may be replaced with full-featured assignable advanced indexing.
//
// `axis` must be within [0, b.ndim()).
// `indices` must have dtype kind of either kInt or kUInt.

Array AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b);

// Takes elements specified by indices from an array.
// Indices that are out of bounds are wrapped around.
//
// `axis` must be within [0, a.ndim()).
// `indices` must have dtype kind of either kInt or kUInt.
//
// TODO(niboshi): Support Scalar and StackVector as indices.
// TODO(niboshi): Support axis=None behavior in NumPy.

Array Take(const Array& a, const Array& indices, int8_t axis);

}  // namespace chainerx
