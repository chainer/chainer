#pragma once

#include <pybind11/pybind11.h>

#include "xchainer/shape.h"

namespace xchainer {
namespace python {
namespace python_internal {

Shape ToShape(const pybind11::tuple& tup);

pybind11::tuple ToTuple(const Shape& shape);

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
