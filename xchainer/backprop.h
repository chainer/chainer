#pragma once

#include <functional>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/constant.h"

namespace xchainer {

enum class DoubleBackpropOption : bool {
    kDisable = false,
    kEnable = true,
};

void Backward(const Array& output, const GraphId& graph_id = kDefaultGraphId,
              DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

void Backward(const std::vector<Array>& outputs, const GraphId& graph_id = kDefaultGraphId,
              DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

void Backward(const std::vector<ConstArrayRef>& outputs, const GraphId& graph_id = kDefaultGraphId,
              DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

}  // namespace xchainer
