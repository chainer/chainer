#pragma once

#include <pybind11/pybind11.h>

#include "xchainer/strides.h"

namespace xchainer {
namespace python {
namespace python_internal {

Strides ToStrides(const pybind11::tuple& tup);

pybind11::tuple ToTuple(const Strides& strides);

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
