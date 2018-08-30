#pragma once

#include "xchainer/array.h"
#include "xchainer/axes.h"

namespace xchainer {
namespace native {

Array TensorDot(const Array& a, const Array& b, const Axes& a_axis, const Axes& b_axis);

}  // namespace native
}  // namespace xchainer
