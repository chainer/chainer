#include "chainerx/python/cuda/cuda_module.h"

#include <pybind11/pybind11.h>

#include "chainerx/python/cuda/memory_pool.h"

namespace chainerx {
namespace python {
namespace cuda {
namespace cuda_internal {

namespace py = pybind11;

void InitChainerxCudaModule(py::module& m) { InitChainerxMemoryPool(m); }

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace python
}  // namespace chainerx
