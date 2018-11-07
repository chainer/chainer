#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace cuda {
namespace cuda_internal {

class MemoryPoolTest;  // for unit-tests

}  // namespace cuda_internal

constexpr size_t kAllocationUnitSize = 512;

enum class MallocStatus { kSuccess = 0, kErrorMemoryAllocation };

class OutOfMemoryError : public ChainerxError {
public:
    explicit OutOfMemoryError(size_t bytesize) : ChainerxError{"Out of memory allocating ", bytesize, " bytes."} {}
};

// TODO(hvy): Add a member function to check for the last error, using e.g. cudaPeekAtLastError.
// This function may for instance throw in case the return value is not a cudaSuccess.
// This will be necessary when extending the MemoryPool with a function to explicitly free blocks.
class Allocator {
public:
    // Allocates memory.
    // This function may throw.
    virtual MallocStatus Malloc(void** ptr, size_t bytesize) = 0;

    // Frees allocated memory.
    // This function does not throw, since it should be usable from within a destructor.
    virtual void Free(void* ptr) = 0;
};

class DeviceMemoryAllocator : public Allocator {
public:
    MallocStatus Malloc(void** ptr, size_t bytesize) override;
    void Free(void* ptr) override { cudaFree(ptr); }
};

class PinnedMemoryAllocator : public Allocator {
public:
    MallocStatus Malloc(void** ptr, size_t bytesize) override;
    void Free(void* ptr) override { cudaFreeHost(ptr); }
};

// Memory pool base.
// This class is thread safe.
class MemoryPool {
public:
    explicit MemoryPool(int device_index, std::unique_ptr<Allocator> allocator)
        : device_index_{device_index}, allocator_{std::move(allocator)} {}

    ~MemoryPool();

    void FreeAllBlocks();

    void* Malloc(size_t bytesize);

    void Free(void* ptr);

private:
    friend class cuda_internal::MemoryPoolTest;  // for unit-tests

    int device_index_;
    std::unique_ptr<Allocator> allocator_;
    std::unordered_map<void*, size_t> in_use_;
    std::vector<std::vector<void*>> free_bins_;
    std::mutex in_use_mutex_;
    std::mutex free_bins_mutex_;
};

}  // namespace cuda
}  // namespace chainerx
