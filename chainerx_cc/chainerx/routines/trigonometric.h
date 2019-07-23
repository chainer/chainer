#pragma once

#include "chainerx/array.h"

namespace chainerx {

Array Sin(const Array& x);

Array Cos(const Array& x);

Array Tan(const Array& x);

Array Arcsin(const Array& x);

Array Arccos(const Array& x);

Array Arctan(const Array& x);

Array Arctan2(const Array& x1, const Array& x2);

}  // namespace chainerx
