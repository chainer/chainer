#pragma once

#include <memory>
#include <vector>

#include "xchainer/array.h"

namespace xchainer {
namespace gradient_internal {

using Arrays = std::vector<Array>;

Arrays CalculateNumericalGradient(std::function<Arrays(const Arrays&)> func, const Arrays& inputs, const Arrays& grad_outputs,
                                  const Arrays& eps);

}  // namespace gradient_internal

using gradient_internal::CalculateNumericalGradient;

}  // namespce xchainer
