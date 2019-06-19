#pragma once

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

Array AbsoluteError(const Array& x1, const Array& x2);

Array SquaredError(const Array& x1, const Array& x2);

Array GaussianKLDivergence(const Array& mean, const Array& ln_var);

Array HuberLoss(const Array& x1, const Array& x2, Scalar delta);

}  // namespace chainerx
