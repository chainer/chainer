#pragma once

#include <pybind11/pybind11.h>

namespace xchainer {

void InitXchainerDevice(pybind11::module&);

}  // namespace xchainer
