#pragma once

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

Array MeanAbsoluteError(const Array& x1, const Array& x2);

Array MeanSquaredError(const Array& x1, const Array& x2);

Array GaussianKLDivergence(const Array& mean, const Array& ln_var);

Array HuberLoss(const Array& x1, const Array& x2, Scalar delta);

}  // namespace chainerx
