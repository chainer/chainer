#pragma once

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/dtype.h"

namespace chainerx {

Array Dot(const Array& a, const Array& b, nonstd::optional<Dtype> out_dtype = nonstd::nullopt);

Array Solve(const Array& a, const Array& b);

Array Inverse(const Array& a);

}  // namespace chainerx
