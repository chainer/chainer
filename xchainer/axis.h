#pragma once

#include <cstdint>

#include "xchainer/ndim_vector.h"

namespace xchainer {
namespace internal {

bool IsAxesPermutation(const NdimVector<int8_t>& axes, int8_t ndim);

}  // namespace internal
}  // namespace xchainer
