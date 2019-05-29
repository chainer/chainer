#pragma once

#include <cstdint>
#include <string>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"

namespace chainerx {

Array MeanAbsoluteError(const Array& x1, const Array& x2);

Array MeanSquaredError(const Array& x1, const Array& x2);

Array GaussianKLDivergence(const Array& mu, const Array& ln_var);

Array HuberLoss(const Array& x1, const Array& x2, Scalar delta, const std::string& reduce);

}  // namespace chainerx
