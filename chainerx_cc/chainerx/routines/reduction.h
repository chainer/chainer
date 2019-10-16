#pragma once

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"

namespace chainerx {

Array Sum(const Array& a, const OptionalAxes& axis = absl::nullopt, bool keepdims = false);

// Returns the LogSumExp (LSE) of x, reduced along the specified axes.
// If no axes are specified, all axes will be reduced.
Array LogSumExp(const Array& x, const OptionalAxes& axis = absl::nullopt, bool keepdims = false);

// Returns the logarithm of the softmax of x along the specified axes.
// If no axes are specified, the softmax is applied on the second axis.
Array LogSoftmax(const Array& x, const OptionalAxes& axis = absl::nullopt);

Array Softmax(const Array& x, const OptionalAxes& axis = absl::nullopt);

Array Cumsum(const Array& a, absl::optional<int8_t> axis = absl::nullopt);

Array Nansum(const Array& a, const OptionalAxes& axis = absl::nullopt, bool keepdims = false);

}  // namespace chainerx
