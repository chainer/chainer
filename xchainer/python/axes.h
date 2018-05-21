#pragma once

#include <pybind11/pybind11.h>

#include "xchainer/axes.h"

namespace xchainer {
namespace python {
namespace internal {

Axes ToAxes(const pybind11::tuple& tup);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
