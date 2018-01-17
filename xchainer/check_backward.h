#pragma once

#include <functional>
#include <vector>

#include "xchainer/array.h"

namespace xchainer {

using Arrays = std::vector<Array>;

void CheckBackwardComputation(std::function<Arrays(const Arrays&)> func, const Arrays& inputs, const Arrays& grad_outputs,
                              const Arrays& eps, float atol = 1e-5, float rtol = 1e-4);

}  // namespace xchainer
