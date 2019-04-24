#pragma once

#include <pybind11/pybind11.h>

namespace chainerx {
namespace python {
namespace python_internal {

void InitChainerxCheckBackward(pybind11::module& m);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
