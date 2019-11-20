#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "chainerx/shape.h"

namespace chainerx {
namespace python {
namespace python_internal {

Shape ToShape(pybind11::handle shape);

pybind11::tuple ToTuple(const Shape& shape);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
