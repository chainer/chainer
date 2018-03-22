#pragma once

#include <pybind11/pybind11.h>

namespace xchainer {
namespace python {
namespace internal {

void InitXchainerScalar(pybind11::module&);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
