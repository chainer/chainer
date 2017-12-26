#pragma once

#include <functional>

#include "array.h"

namespace xchainer {

using Arrays = std::vector<Array>;
using ForwardFunction = std::function<std::vector<Array>(const std::vector<Array>&)>;

void CheckBackwardComputation(const ForwardFunction& func, const Arrays& inputs, const Arrays& grad_outputs, const Arrays& eps,
                              float atol = 1e-5, float rtol = 1e-4);

}  // namespace xchainer
