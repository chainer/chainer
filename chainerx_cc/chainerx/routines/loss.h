#pragma once

#include <cstdint>
#include <string>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"

namespace chainerx {

Array MeanAbsoluteError(const Array& x0, const Array& x1);

Array MeanSquaredError(const Array& x0, const Array& x1);

Array GaussianKLDivergence(const Array& mu, const Array& ln_var, const std::string& reduction);

}  // namespace chainerx
