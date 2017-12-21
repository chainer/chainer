#pragma once

#include <pybind11/pybind11.h>

namespace xchainer {
namespace cuda {

void InitXchainerCudaHello(pybind11::module&);

}  // namespace cuda
}  // namespace xchainer
