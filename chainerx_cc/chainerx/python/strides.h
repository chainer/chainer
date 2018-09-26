#pragma once

#include <pybind11/pybind11.h>

#include "chainerx/strides.h"

namespace chainerx {
namespace python {
namespace python_internal {

Strides ToStrides(const pybind11::tuple& tup);

pybind11::tuple ToTuple(const Strides& strides);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
