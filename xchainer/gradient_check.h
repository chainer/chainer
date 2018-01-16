#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "xchainer/array.h"

namespace xchainer {
namespace gradient_internal {

using Arrays = std::vector<Array>;
using ForwardFunction = std::function<std::vector<Array>(const std::vector<Array>&)>;

Arrays CalculateNumericalGradient(std::function<Arrays(const Arrays&)> func, const Arrays& inputs, const Arrays& grad_outputs,
                                  const Arrays& eps);

void CheckBackwardComputation(const ForwardFunction& func, const Arrays& inputs, const Arrays& grad_outputs, const Arrays& eps,
                              float atol = 1e-5, float rtol = 1e-4);

}  // namespace gradient_internal

using gradient_internal::CalculateNumericalGradient;
using gradient_internal::CheckBackwardComputation;

}  // namespce xchainer
