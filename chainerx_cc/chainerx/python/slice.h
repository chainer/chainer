#pragma once

#include <pybind11/pybind11.h>

#include "chainerx/slice.h"

namespace chainerx {
namespace python {
namespace python_internal {

Slice MakeSlice(const pybind11::slice& slice);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
