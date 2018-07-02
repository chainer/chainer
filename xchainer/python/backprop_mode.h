#pragma once

#include <pybind11/pybind11.h>

#include "xchainer/backprop_mode.h"

namespace xchainer {
namespace python {
namespace internal {

void InitXchainerBackpropMode(pybind11::module&);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
