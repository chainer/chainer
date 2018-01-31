#pragma once

#include "xchainer/array.h"

namespace xchainer {

struct LeaveGraphTag {};
constexpr LeaveGraphTag kLeaveGraph{};

void Backward(Array& output, const GraphId& graph_id = "");
void Backward(LeaveGraphTag, Array& output, const GraphId& graph_id = "");

}  // namespace xchainer
