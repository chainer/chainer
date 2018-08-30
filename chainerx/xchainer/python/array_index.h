#pragma once

#include <vector>

#include <pybind11/pybind11.h>

#include "xchainer/array_index.h"

namespace xchainer {
namespace python {
namespace python_internal {

namespace py = pybind11;

std::vector<ArrayIndex> MakeArrayIndices(py::handle handle);

void InitXchainerArrayIndex(py::module&);

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
