#pragma once

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/dtype.h"

namespace chainerx {

Array Dot(const Array& a, const Array& b, nonstd::optional<Dtype> out_dtype = nonstd::nullopt);

Array Trace(
        const Array& x,
        const int64_t offset = 0,
        const int64_t axis1 = 0,
        const int64_t axis2 = 1,
        nonstd::optional<Dtype> dtype = nonstd::nullopt);

}  // namespace chainerx
