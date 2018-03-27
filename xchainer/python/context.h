#pragma once

#include <pybind11/pybind11.h>

namespace xchainer {
namespace python {
namespace internal {

void InitXchainerContext(pybind11::module&);

void InitXchainerContextScope(pybind11::module&);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
