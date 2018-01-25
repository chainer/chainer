#pragma once

#include "xchainer/array.h"

namespace xchainer {

void Backward(Array& output, const GraphId& graph_id = "");

}  // namespace xchainer
