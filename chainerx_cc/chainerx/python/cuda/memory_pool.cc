#include "chainerx/python/cuda/memory_pool.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <pybind11/functional.h>

#include "chainerx/cuda/cuda_device.h"
#include "chainerx/cuda/memory_pool.h"
#include "chainerx/device.h"

#include "chainerx/python/backend.h"
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

std::vector<intptr_t> GetMemoryPools() {
    Backend& backend = GetBackend("cuda");
    size_t device_count = static_cast<size_t>(backend.GetDeviceCount());

    std::vector<intptr_t> memory_pools;
    memory_pools.reserve(device_count);

    for (size_t device_id = 0; device_id < device_count; ++device_id) {
        CudaDevice* device = dynamic_cast<CudaDevice*>(&backend.GetDevice(device_id));
        const std::shared_ptr<MemoryPool>& memory_pool = device->device_memory_pool();
        memory_pools.emplace_back(reinterpret_cast<intptr_t>(memory_pool.get()));  // NOLINT(cppcoreguidelines-cppcoreguidelines);
    }
    return memory_pools;
}

std::pair<intptr_t, intptr_t> GetMemoryPoolMallocFree() {
    return std::make_pair(
            reinterpret_cast<intptr_t>(&Malloc), reinterpret_cast<intptr_t>(&Free));  // NOLINT(cppcoreguidelines-cppcoreguidelines);
}

void InitChainerxMemoryPool(py::module& m) {
    m.def("get_memory_pools", []() -> std::vector<intptr_t> { return GetMemoryPools(); });
    m.def("get_memory_pool_malloc_free", []() -> std::pair<intptr_t, intptr_t> { return GetMemoryPoolMallocFree(); });
}

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace python
}  // namespace chainerx
