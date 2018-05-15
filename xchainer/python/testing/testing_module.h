#pragma once

#include <pybind11/pybind11.h>

namespace xchainer {
namespace python {
namespace testing {
namespace internal {

void InitXchainerTestingModule(pybind11::module&);

}  // namespace internal
}  // namespace testing
}  // namespace python
}  // namespace xchainer
