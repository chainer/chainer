#pragma once

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

Array ClippedRelu(const Array& x, Scalar z);

Array Crelu(const Array& x, nonstd::optional<int8_t> axis);

Array Elu(const Array& x, Scalar alpha);

Array Sigmoid(const Array& x);

Array Relu(const Array& x);

Array LeakyRelu(const Array& x, Scalar slope);

}  // namespace chainerx
