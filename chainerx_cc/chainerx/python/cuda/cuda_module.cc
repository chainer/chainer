#include "chainerx/python/cuda/cuda_module.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include <pybind11/pybind11.h>

#include "chainerx/backend.h"
#include "chainerx/context.h"
#include "chainerx/cuda/cuda_device.h"
#include "chainerx/cuda/memory_pool.h"

#include "chainerx/python/common.h"

namespace chainerx {
namespace python {
namespace cuda {
namespace cuda_internal {

namespace py = pybind11;

using CudaDevice = chainerx::cuda::CudaDevice;

void* Malloc(void* backend, size_t bytesize, int device_id) {
    CudaDevice* device = dynamic_cast<CudaDevice*>(
            &reinterpret_cast<Backend*>(backend)->GetDevice(device_id));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    return device->device_memory_pool()->Malloc(bytesize);
}

void Free(void* backend, void* ptr, int device_id) {
    CudaDevice* device = dynamic_cast<CudaDevice*>(
            &reinterpret_cast<Backend*>(backend)->GetDevice(device_id));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    return device->device_memory_pool()->Free(ptr);
}

intptr_t GetBackendPtr() {
    return reinterpret_cast<intptr_t>(&GetBackend("cuda"));  // NOLINT(cppcoreguidelines-cppcoreguidelines);
}

std::pair<intptr_t, intptr_t> GetBackendMallocFreePtrs() {
    return std::make_pair(
            reinterpret_cast<intptr_t>(&Malloc), reinterpret_cast<intptr_t>(&Free));  // NOLINT(cppcoreguidelines-cppcoreguidelines);
}

void InitChainerxCudaModule(py::module& m) {
    // Exposes pointers to objects and functions so that external sources can allocate CUDA memory via ChainerX.
    m.def("get_backend_ptr", []() -> intptr_t { return GetBackendPtr(); });
    m.def("get_backend_malloc_free_ptrs", []() -> std::pair<intptr_t, intptr_t> { return GetBackendMallocFreePtrs(); });
}

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace python
}  // namespace chainerx
