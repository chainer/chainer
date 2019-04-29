#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"

namespace chainerx {

Array MeanAbsoluteError(const Array& y, const Array& targ);

Array MeanSquaredError(const Array& y, const Array& targ);

Array GaussianKLDivergence(const Array& mu, const Array& ln_var, const std::string& reduction);

}

