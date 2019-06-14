#pragma once

#include <memory>
#include <vector>

#include "chainerx/array.h"

namespace chainerx {
namespace numerical_gradient_internal {

using Arrays = std::vector<Array>;

Arrays CalculateNumericalGradient(
        std::function<Arrays(const Arrays&)> func, const Arrays& inputs, const Arrays& grad_outputs, const Arrays& eps);

}  // namespace numerical_gradient_internal

using numerical_gradient_internal::CalculateNumericalGradient;

}  // namespace chainerx
