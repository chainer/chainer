#pragma once

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"

namespace chainerx {

Array ArgMax(const Array& a, const OptionalAxes& axis = nonstd::nullopt);

}  // namespace chainerx
