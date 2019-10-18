#pragma once

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"

namespace chainerx {

Array ArgMax(const Array& a, const OptionalAxes& axis = absl::nullopt);

Array ArgMin(const Array& a, const OptionalAxes& axis = absl::nullopt);

Array CountNonzero(const Array& a, const OptionalAxes& axis = absl::nullopt);

Array NanArgMax(const Array& a, const OptionalAxes& axis = absl::nullopt);

Array NanArgMin(const Array& a, const OptionalAxes& axis = absl::nullopt);

}  // namespace chainerx
