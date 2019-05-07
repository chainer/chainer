#pragma once

#include <vector>

#include <pybind11/pybind11.h>

#include "chainerx/array_index.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

std::vector<ArrayIndex> MakeArrayIndices(py::handle handle);

void InitChainerxArrayIndex(py::module& m);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
