#include "chainerx/python/common_export.h"

#include "chainerx/python/cuda/cuda_module.h"

#include <cstddef>
#include <cstdint>
#include <tuple>

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

void* Malloc(void* param, size_t bytesize, int device_id) {
    CudaDevice& device = dynamic_cast<CudaDevice&>(
            reinterpret_cast<Backend*>(param)->GetDevice(device_id));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    return device.device_memory_pool()->Malloc(bytesize);
}

void Free(void* param, void* ptr, int device_id) {
    CudaDevice& device = dynamic_cast<CudaDevice&>(
            reinterpret_cast<Backend*>(param)->GetDevice(device_id));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    return device.device_memory_pool()->Free(ptr);
}

std::tuple<intptr_t, intptr_t, intptr_t> GetCAllocator() {
    return std::make_tuple(
            reinterpret_cast<intptr_t>(&GetBackend("cuda")),  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<intptr_t>(&Malloc),  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<intptr_t>(&Free));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

void InitChainerxCudaModule(py::module& m) {
    // Exposes pointers to objects and functions so that external sources can allocate CUDA memory via ChainerX.
    m.def("get_c_allocator", []() -> std::tuple<intptr_t, intptr_t, intptr_t> { return GetCAllocator(); });
}

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace python
}  // namespace chainerx
