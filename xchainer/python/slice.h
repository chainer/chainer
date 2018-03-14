#pragma once

#include <pybind11/pybind11.h>

#include "xchainer/slice.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;

Slice MakeSlice(const py::slice& slice);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
