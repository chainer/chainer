#pragma once

#include "xchainer/array.h"
#include "xchainer/scalar.h"

namespace xchainer {

bool AllClose(const Array& a, const Array& b, double rtol = 1e-5, double atol = 1e-8, bool equal_nan = false);

}  // namespace xchainer
