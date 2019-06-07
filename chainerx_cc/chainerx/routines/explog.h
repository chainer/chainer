#pragma once

#include <cstdint>

#include "chainerx/array.h"

namespace chainerx {

Array Exp(const Array& x);

Array Expm1(const Array& x);

Array Exp2(const Array& x);

Array Log(const Array& x);

Array Log10(const Array& x);

Array Log1p(const Array& x);

}  // namespace chainerx
