#pragma once

#include <vector>

#include <pybind11/pybind11.h>

#include "xchainer/array_index.h"

namespace py = pybind11;

namespace xchainer {
namespace python {
namespace internal {

std::vector<ArrayIndex> MakeArrayIndices(py::handle handle);

void InitXchainerArrayIndex(py::module&);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
