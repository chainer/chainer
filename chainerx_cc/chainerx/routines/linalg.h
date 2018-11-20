#pragma once

#include "chainerx/array.h"

namespace chainerx {

Array Dot(const Array& a, const Array& b);
Array Linear(const Array& x, const Array& w, const nonstd::optional<Array>& b = nonstd::nullopt, uint8_t n_batch_axes = 1);

}  // namespace chainerx
