#pragma once

#include <pybind11/pybind11.h>

#include "chainerx/context.h"

namespace chainerx {
namespace python {
namespace python_internal {

Context& GetContext(pybind11::handle handle);

void InitChainerxContext(pybind11::module& m);

void InitChainerxContextScope(pybind11::module& m);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
