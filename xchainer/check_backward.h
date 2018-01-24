#pragma once

#include <functional>
#include <string>
#include <vector>

#include "xchainer/array.h"

namespace xchainer {

void CheckBackwardComputation(std::function<std::vector<Array>(const std::vector<Array>&)> func, const std::vector<Array>& inputs,
                              const std::vector<Array>& grad_outputs, const std::vector<Array>& eps, double atol = 1e-5,
                              double rtol = 1e-4, const std::string& graph_name = "");

}  // namespace xchainer
