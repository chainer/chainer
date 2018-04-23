#pragma once

#include "xchainer/constant.h"
#include "xchainer/stack_vector.h"

namespace xchainer {

template <typename T>
using NdimVector = StackVector<T, kMaxNdim>;

}  // namespace xchainer
