#pragma once

#include "xchainer/array.h"
#include "xchainer/constant.h"

namespace xchainer {

void Backward(Array& output, const GraphId& graph_id = kDefaultGraphId);

}  // namespace xchainer
