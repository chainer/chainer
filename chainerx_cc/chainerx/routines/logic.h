#pragma once

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"

namespace chainerx {

// Returns an elementwise equality array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array Equal(const Array& x1, const Array& x2);

// Returns an elementwise non-equality array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array NotEqual(const Array& x1, const Array& x2);

// Returns an elementwise `x1` > `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array Greater(const Array& x1, const Array& x2);

// Returns an elementwise `x1` >= `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array GreaterEqual(const Array& x1, const Array& x2);

// Returns an elementwise `x1` < `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
inline Array Less(const Array& x1, const Array& x2) { return Greater(x2, x1); }

// Returns an elementwise `x1` <= `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
inline Array LessEqual(const Array& x1, const Array& x2) { return GreaterEqual(x2, x1); }

// Returns an elementwise logical negation of an array.
Array LogicalNot(const Array& x);

Array LogicalAnd(const Array& x1, const Array& x2);

// TODO(imanishi): Add python binding
Array LogicalAnd(const Array& x1, Scalar x2);

Array LogicalOr(const Array& x1, const Array& x2);

// TODO(imanishi): Add python binding
Array LogicalOr(const Array& x1, Scalar x2);

Array LogicalXor(const Array& x1, const Array& x2);

Array All(const Array& a, const OptionalAxes& axis = absl::nullopt, bool keepdims = false);

Array Any(const Array& a, const OptionalAxes& axis = absl::nullopt, bool keepdims = false);

Array IsNan(const Array& x);

Array IsInf(const Array& x);

Array IsFinite(const Array& x);

}  // namespace chainerx
