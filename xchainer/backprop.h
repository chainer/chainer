#pragma once

#include <string>

#include "xchainer/array.h"

namespace xchainer {

void Backward(Array& output, const std::string& graph_name = "");

}  // namespace xchainer
