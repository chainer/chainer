#pragma once

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"

namespace chainerx {

Array Mean(const Array& a, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);

Array Var(const Array& a, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);

}  // namespace chainerx
