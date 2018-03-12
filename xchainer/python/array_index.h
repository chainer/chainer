#pragma once

#include <vector>

#include <pybind11/pybind11.h>

#include "xchainer/array_index.h"

namespace py = pybind11;

namespace xchainer {
namespace python {
namespace internal {

std::vector<ArrayIndex> MakeArrayIndices(py::handle handle);

}  // namespace internal
}  // namespace python

void InitXchainerArrayIndex(py::module&);

}  // namespace xchainer
