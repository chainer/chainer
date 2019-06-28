#pragma once

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"

namespace chainerx {

Array Sum(const Array& a, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);

// Returns the LogSumExp (LSE) of x, reduced along the specified axes.
// If no axes are specified, all axes will be reduced.
Array LogSumExp(const Array& x, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);

// Returns the logarithm of the softmax of x along the specified axes.
// If no axes are specified, the softmax is applied on the second axis.
Array LogSoftmax(const Array& x, const OptionalAxes& axis = nonstd::nullopt);

Array Softmax(const Array& x, const OptionalAxes& axis = nonstd::nullopt);

}  // namespace chainerx
