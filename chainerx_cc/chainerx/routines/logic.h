#pragma once

#include "chainerx/array.h"

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
Array Less(const Array& x1, const Array& x2);

// Returns an elementwise `x1` <= `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array LessEqual(const Array& x1, const Array& x2);

// Returns an elementwise logical negation of an array.
Array LogicalNot(const Array& x1);

}  // namespace chainerx
