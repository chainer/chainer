#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/scalar.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace native {
namespace native_internal {

Array Im2Col(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all,
        Scalar pad_value = 0);

}  // namespace native_internal
}  // namespace native
}  // namespace chainerx
