#pragma once

#include <pybind11/pybind11.h>

namespace chainerx {
namespace python {
namespace cuda {
namespace cuda_internal {

void InitChainerxCudaModule(pybind11::module& m);

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace python
}  // namespace chainerx
