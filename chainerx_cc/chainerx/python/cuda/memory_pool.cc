#include "chainerx/python/cuda/memory_pool.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include <pybind11/functional.h>

#include "chainerx/cuda/cuda_device.h"
#include "chainerx/cuda/memory_pool.h"
#include "chainerx/device.h"

#include "chainerx/python/common.h"
#include "chainerx/python/device.h"

namespace chainerx {
namespace python {
namespace cuda {
namespace cuda_internal {

namespace py = pybind11;  // standard convention

void InitChainerxMemoryPool(py::module& m) {
    m.def("get_memory_pool_malloc",
          [](py::handle device) -> std::function<intptr_t(size_t)> {
              Device& actual_device = chainerx::python::python_internal::GetDevice(device);

              std::function<void*(size_t)> malloc = std::bind(
                      &chainerx::cuda::MemoryPool::Malloc,
                      dynamic_cast<chainerx::cuda::CudaDevice*>(&actual_device)->device_memory_pool().get(),
                      std::placeholders::_1);

              return [malloc = std::move(malloc)](size_t bytesize) {
                  return reinterpret_cast<intptr_t>(malloc(bytesize));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
              };
          },
          py::arg("device"));
    m.def("get_memory_pool_free",
          [](py::handle device) -> std::function<void(intptr_t)> {
              Device& actual_device = chainerx::python::python_internal::GetDevice(device);

              std::function<void(void*)> free = std::bind(
                      &chainerx::cuda::MemoryPool::Free,
                      dynamic_cast<chainerx::cuda::CudaDevice*>(&actual_device)->device_memory_pool().get(),
                      std::placeholders::_1);

              return [free = std::move(free)](intptr_t bytesize) {
                  free(reinterpret_cast<void*>(bytesize));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
              };
          },
          py::arg("device"));
}

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace python
}  // namespace chainerx
