#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "xchainer/array.h"

namespace xchainer {
namespace gradient_internal {

using Arrays = std::vector<Array>;

Arrays CalculateNumericalGradient(std::function<Arrays(const Arrays&)> func, const Arrays& inputs, const Arrays& grad_outputs,
                                  const Arrays& eps);

void CheckBackwardComputation(std::function<Arrays(const Arrays&)> func, const Arrays& inputs, const Arrays& grad_outputs,
                              const Arrays& eps, float atol = 1e-5, float rtol = 1e-4);

}  // namespace gradient_internal

using gradient_internal::CalculateNumericalGradient;
using gradient_internal::CheckBackwardComputation;

}  // namespace xchainer
