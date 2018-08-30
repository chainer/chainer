#pragma once

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/axes.h"

namespace xchainer {

Array Mean(const Array& a, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);

Array Var(const Array& a, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);

}  // namespace xchainer
