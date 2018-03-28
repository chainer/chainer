#pragma once

#include "xchainer/array.h"

namespace xchainer {

// Returns an elementwise equality array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array Equal(const Array& x1, const Array& x2);

}  // namespace xchainer
