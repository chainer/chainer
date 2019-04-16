#pragma once

#include <cstdint>
#include <vector>

#include "nonstd/optional.hpp"

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/backend.h"
#include "chainerx/device.h"
#include "chainerx/op.h"

namespace chainerx {

class AddAtOp : public Op {
public:
    static const char* name() { return "AddAt"; }

    virtual void Call(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) = 0;
};

class TakeOp : public Op {
public:
    static const char* name() { return "Take"; }

    virtual void Call(const Array& a, const Array& indices, int8_t axis, const Array& out) = 0;
};

class DiagonalOp : public Op {
public:
    static const char* name() { return "Diagonal"; }
    virtual void Call(const Array& x, int64_t offset, int64_t axis1, int64_t axis2, Array& out) = 0;
};

namespace internal {

// Returns a view selected with the indices.
Array At(const Array& a, const std::vector<ArrayIndex>& indices);

}  // namespace internal

// Adds each slice of `b` along the axis `axis` to `a`'s corresponding slices, specified by `indices`.
// The result is assigned in `out. Input arrays `a`, `indices`, and `b` are not altered.
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

Array Diagonal(const Array& x, const int64_t offset = 0, const int64_t axis1 = 0, const int64_t axis2 = 1);

}  // namespace chainerx
