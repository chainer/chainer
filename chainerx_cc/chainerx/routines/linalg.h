#pragma once

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/dtype.h"

namespace chainerx {

Array Dot(const Array& a, const Array& b, absl::optional<Dtype> out_dtype = absl::nullopt);

Array Solve(const Array& a, const Array& b);

Array Inverse(const Array& a);

Array Cholesky(const Array& a);

}  // namespace chainerx
