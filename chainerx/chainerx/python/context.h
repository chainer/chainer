#pragma once

#include <pybind11/pybind11.h>

#include "chainerx/context.h"

namespace chainerx {
namespace python {
namespace python_internal {

Context& GetContext(pybind11::handle handle);

void InitXchainerContext(pybind11::module&);

void InitXchainerContextScope(pybind11::module&);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
