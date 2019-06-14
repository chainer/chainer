#pragma once

#include <pybind11/pybind11.h>

namespace chainerx {
namespace python {
namespace testing {
namespace testing_internal {

void InitChainerxTestingModule(pybind11::module& m);

}  // namespace testing_internal
}  // namespace testing
}  // namespace python
}  // namespace chainerx
