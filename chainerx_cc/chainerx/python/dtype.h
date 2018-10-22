#pragma once

#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "chainerx/dtype.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention

Dtype GetDtypeFromString(const std::string& name);

Dtype GetDtypeFromNumpyDtype(const py::dtype& npdtype);

Dtype GetDtype(pybind11::handle handle);

py::dtype GetNumpyDtype(Dtype dtype);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
