#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/dims.h"

namespace chainerx {
namespace native {
namespace native_internal {

Array Col2Im(const Array& col, const Dims& stride, const Dims& pad, const Dims& out_size);

}  // namespace native_internal
}  // namespace native
}  // namespace chainerx
