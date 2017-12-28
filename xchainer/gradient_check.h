#pragma once

#include <memory>
#include <vector>

#include "xchainer/array.h"

namespace xchainer {
namespace gradient_detail {

using Arrays = std::vector<Array>;

Arrays CalculateNumericalGradient(std::function<Arrays(const Arrays&)> func, const Arrays& inputs, const Arrays& grad_outputs,
                                  const Arrays& eps);

}  // namespace gradient_detail

using gradient_detail::CalculateNumericalGradient;

}  // namespce xchainer
