#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/dims.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace native {
namespace native_internal {

Array Im2Col(const Array& x, const Dims& kernel_size, const Dims& stride, const Dims& pad, bool cover_all, Scalar pad_value = 0);

}  // namespace native_internal
}  // namespace native
}  // namespace chainerx
