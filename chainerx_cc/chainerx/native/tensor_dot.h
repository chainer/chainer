#pragma once

#include "chainerx/array.h"
#include "chainerx/axes.h"

namespace chainerx {
namespace native {

Array TensorDot(const Array& a, const Array& b, const Axes& a_axis, const Axes& b_axis, Dtype out_dtype);

}  // namespace native
}  // namespace chainerx
