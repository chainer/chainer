#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace native {
namespace native_internal {

Array Col2Im(
        const Array& col,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& out_size);

}  // namespace native_internal
}  // namespace native
}  // namespace chainerx
