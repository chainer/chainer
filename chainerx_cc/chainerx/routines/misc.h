#pragma once

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

Array Sqrt(const Array& x);

Array Square(const Array& x);

Array SquaredDifference(const Array& x1, const Array& x2);

Array Absolute(const Array& x);

Array Fabs(const Array& x);

Array Sign(const Array& x);

Array Maximum(const Array& x1, Scalar x2);
Array Maximum(Scalar x1, const Array& x2);
Array Maximum(const Array& x1, const Array& x2);

Array Minimum(const Array& x1, Scalar x2);
Array Minimum(Scalar x1, const Array& x2);
Array Minimum(const Array& x1, const Array& x2);

Array Clip(const Array& a, Scalar a_min, Scalar a_max);

}  // namespace chainerx
