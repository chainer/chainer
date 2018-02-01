#pragma once

#include "xchainer/array.h"

namespace xchainer {

enum class DoubleBackpropOption : bool {
    kDisable = false,
    kEnable = true,
};

void Backward(Array& output, const GraphId& graph_id = "", DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

}  // namespace xchainer
