#pragma once

#include <vector>

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

Array ClippedRelu(const Array& x, Scalar z);

Array CRelu(const Array& x, int8_t axis);

Array Elu(const Array& x, double alpha = 1.0);

Array Sigmoid(const Array& x);

Array Relu(const Array& x);

Array LeakyRelu(const Array& x, Scalar slope);

std::vector<Array> TreeLstm(std::vector<Array> arrays);

std::vector<Array> SLstm(const Array& c_prev1, const Array& c_prev2, const Array& x1, const Array& x2);

Array Softplus(const Array& x, double beta = 1.0);

}  // namespace chainerx
