#pragma once

#include <pybind11/pybind11.h>

#include "chainerx/backprop_mode.h"

namespace chainerx {
namespace python {
namespace python_internal {

void InitChainerxBackpropMode(pybind11::module& m);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
