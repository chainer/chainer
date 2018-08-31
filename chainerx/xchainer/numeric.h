#pragma once

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

bool AllClose(const Array& a, const Array& b, double rtol = 1e-5, double atol = 1e-8, bool equal_nan = false);

}  // namespace chainerx
