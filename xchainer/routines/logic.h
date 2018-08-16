#pragma once

#include "xchainer/array.h"

namespace xchainer {

// Returns an elementwise equality array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array Equal(const Array& x1, const Array& x2);

// Returns an elementwise `x1` > `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array Greater(const Array& x1, const Array& x2);

// Returns an elementwise logical negation of an array.
Array Not(const Array& x1);

}  // namespace xchainer
