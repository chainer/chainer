#pragma once

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/scalar.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
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
}  // namespace xchainer
