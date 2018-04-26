#pragma once

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/axes.h"

namespace xchainer {

Array ArgMax(const Array& a, const OptionalAxes& axis = nonstd::nullopt);

}  // namespace xchainer
