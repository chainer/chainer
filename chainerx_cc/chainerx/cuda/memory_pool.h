#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace cuda {

constexpr size_t kAllocationUnitSize = 512;

class Allocator {
public:
    virtual void Malloc(void** ptr, size_t bytesize) = 0;
    virtual void Free(void* ptr) = 0;
};

class DeviceMemoryAllocator : public Allocator {
public:
    void Malloc(void** ptr, size_t bytesize) override { CheckCudaError(cudaMallocManaged(ptr, bytesize, cudaMemAttachGlobal)); }
    void Free(void* ptr) override { cudaFree(ptr); }
};

class PinnedMemoryAllocator : public Allocator {
public:
    void Malloc(void** ptr, size_t bytesize) override { CheckCudaError(cudaHostAlloc(ptr, bytesize, cudaHostAllocWriteCombined)); }
    void Free(void* ptr) override { cudaFreeHost(ptr); }
};

// Memory pool base.
// This class is thread safe.
class MemoryPool {
public:
    explicit MemoryPool(int device_index, std::unique_ptr<Allocator> allocator)
        : device_index_{device_index}, allocator_{std::move(allocator)} {}

    ~MemoryPool();

    void* Malloc(size_t bytesize);

    void Free(void* ptr);

private:
    int device_index_;
    std::unique_ptr<Allocator> allocator_;
    std::unordered_map<void*, size_t> in_use_;
    std::vector<std::vector<void*>> free_bins_;
    std::mutex in_use_mutex_;
    std::mutex free_bins_mutex_;
};

}  // namespace cuda
}  // namespace chainerx
