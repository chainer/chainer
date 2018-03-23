#pragma once

#include <pybind11/pybind11.h>

#include "xchainer/slice.h"

namespace xchainer {
namespace python {
namespace internal {

Slice MakeSlice(const pybind11::slice& slice);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
