#include "chainerx/python/cuda/memory_pool.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include <pybind11/functional.h>

#include "chainerx/cuda/cuda_device.h"
#include "chainerx/cuda/memory_pool.h"
#include "chainerx/device.h"

#include "chainerx/python/common.h"
#include "chainerx/python/context.h"
#include "chainerx/python/device.h"

namespace chainerx {
namespace python {
namespace cuda {
namespace cuda_internal {

namespace py = pybind11;  // standard convention

using CudaDevice = chainerx::cuda::CudaDevice;
using MemoryPool = chainerx::cuda::MemoryPool;

void* Malloc(void* memory_pool_ptr, size_t bytesize) {
    return reinterpret_cast<MemoryPool*>(memory_pool_ptr)->Malloc(bytesize);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

void Free(void* memory_pool_ptr, void* ptr) {
    reinterpret_cast<MemoryPool*>(memory_pool_ptr)->Free(ptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

intptr_t GetMemoryPool(CudaDevice& device) {
    return reinterpret_cast<intptr_t>(device.device_memory_pool().get());  // NOLINT(cppcoreguidelines-cppcoreguidelines);
}

std::pair<intptr_t, intptr_t> GetMallocFree() {
    return std::make_pair(reinterpret_cast<intptr_t>(&Malloc), reinterpret_cast<intptr_t>(&Free)); // NOLINT(cppcoreguidelines-cppcoreguidelines);
}

void InitChainerxMemoryPool(py::module& m) {
    m.def("get_memory_pool",
          [](size_t device_id) -> intptr_t {
              Device& device = GetBackend("cuda").GetDevice(device_id);
              CudaDevice* cuda_device = dynamic_cast<CudaDevice*>(&device);
              return GetMemoryPool(*cuda_device);
          },
          py::arg("device_id"));
    m.def("get_memory_pool_malloc_free", []() -> std::pair<intptr_t, intptr_t> { return GetMallocFree(); });
}

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace python
}  // namespace chainerx
