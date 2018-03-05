#pragma once

#include <functional>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/constant.h"

namespace xchainer {

void CheckBackwardComputation(std::function<std::vector<Array>(const std::vector<Array>&)> func, const std::vector<Array>& inputs,
                              const std::vector<Array>& grad_outputs, const std::vector<Array>& eps, double atol = 1e-5, double rtol = 1e-4,
                              const GraphId& graph_id = kDefaultGraphId);
void CheckDoubleBackwardComputation(std::function<std::vector<Array>(const std::vector<Array>&)> func, const std::vector<Array>& inputs,
                                    const std::vector<Array>& grad_outputs, const std::vector<Array>& grad_grad_inputs,
                                    const std::vector<Array>& eps, double atol = 1e-5, double rtol = 1e-4,
                                    const GraphId& graph_id = kDefaultGraphId);

}  // namespace xchainer
