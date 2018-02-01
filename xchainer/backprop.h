#pragma once

#include "xchainer/array.h"
#include "xchainer/constant.h"

namespace xchainer {

enum class DoubleBackpropOption : bool {
    kDisable = false,
    kEnable = true,
};

void Backward(Array& output, const GraphId& graph_id = kDefaultGraphId,
              DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

}  // namespace xchainer
