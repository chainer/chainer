#pragma once

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

Array ClippedRelu(const Array& x, Scalar z);

Array CRelu(const Array& x, int8_t axis);

Array Elu(const Array& x, float_t alpha = 1.0);

Array Sigmoid(const Array& x);

Array Relu(const Array& x);

Array LeakyRelu(const Array& x, Scalar slope);

}  // namespace chainerx
