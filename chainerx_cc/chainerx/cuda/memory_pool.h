#pragma once

#include <map>
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
constexpr size_t kCompactionThreashold = 512;

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
    // This function must not throw, since it should be usable from within a destructor.
    virtual void Free(void* ptr) noexcept = 0;
};

class DeviceMemoryAllocator : public Allocator {
public:
    MallocStatus Malloc(void** ptr, size_t bytesize) override;
    void Free(void* ptr) noexcept override { cudaFree(ptr); }
};

class PinnedMemoryAllocator : public Allocator {
public:
    MallocStatus Malloc(void** ptr, size_t bytesize) override;
    void Free(void* ptr) noexcept override { cudaFreeHost(ptr); }
};

// A chunk points to a device memory.
//
// A chunk might be a splitted memory block from a larger allocation.
// The prev/next pointers contruct a doubly-linked list of memory addresses
// sorted by base address that must be contiguous.
class Chunk {
public:
    Chunk(void* ptr, size_t offset, size_t bytesize)
        : ptr_{reinterpret_cast<void*>(reinterpret_cast<intptr_t>(ptr) + offset)}, bytesize_(bytesize) {}
    Chunk(const Chunk&) = default;
    ~Chunk() {}

    // Splits this chunk to a chunk of a given bytesize and a chunk of the remaining.
    //
    // Modifies the bytesize and next of this, and returns the remaining.
    std::unique_ptr<Chunk> Split(size_t bytesize);

    // Merges with the next chunk
    void MergeWithNext();

    void SetPrev(Chunk* prev) { prev_ = prev; }
    void SetNext(Chunk* next) { next_ = next; }

    void* ptr() const { return ptr_; }
    size_t bytesize() const { return bytesize_; }
    const Chunk* prev() const { return prev_; }
    Chunk* prev() { return prev_; }
    const Chunk* next() const { return next_; }
    Chunk* next() { return next_; }

private:
    void* ptr_{nullptr};  // Memory address.
    size_t bytesize_{0};  // Chunk bytesize.
    Chunk* prev_{nullptr};  // Prev memory pointer if splitted from a larger allocation
    Chunk* next_{nullptr};  // Next memory pointer if splitted from a larger allocation
};

using FreeList = std::vector<std::unique_ptr<Chunk>>;  // List of free chunks w.r.t. same sizes

// Memory pool.
// This class is thread safe.
class MemoryPool {
public:
    explicit MemoryPool(int device_index, std::unique_ptr<Allocator> allocator)
        : device_index_{device_index}, allocator_{std::move(allocator)} {}

    MemoryPool(const MemoryPool&) = delete;

    ~MemoryPool();

    void FreeUnusedBlocks();

    void* Malloc(size_t bytesize);

    // ChainerxError is thrown if ptr is not an in-use memory pointer.
    void Free(void* ptr);

    void FreeNoExcept(void* ptr) noexcept;

private:
    friend class cuda_internal::MemoryPoolTest;  // for unit-tests

    // Rounds up the memory size to fit memory alignment of cudaMalloc.
    size_t GetAllocationSize(size_t bytesize) { return ((bytesize + kAllocationUnitSize - 1) / kAllocationUnitSize) * kAllocationUnitSize; }
    void PushIntoFreeList(std::unique_ptr<Chunk> chunk);
    std::unique_ptr<Chunk> PopFromFreeList(size_t allocation_size);
    std::unique_ptr<Chunk> RemoveChunkFromFreeList(Chunk* chunk);
    void CompactFreeBins(std::map<size_t, FreeList>::iterator it_start, std::map<size_t, FreeList>::iterator it_end);

    int device_index_;
    std::unique_ptr<Allocator> allocator_;
    std::unordered_map<void*, std::unique_ptr<Chunk>> in_use_;  // ptr => Chunk
    std::map<size_t, FreeList> free_bins_;  // allocation size => FreeList
    std::mutex in_use_mutex_;
    std::mutex free_bins_mutex_;
};

}  // namespace cuda
}  // namespace chainerx
