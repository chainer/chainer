#pragma once

#include "xchainer/array.h"
#include "xchainer/scalar.h"

namespace xchainer {

bool AllClose(const Array& a, const Array& b, double rtol, double atol);

}  // namespace xchainer
