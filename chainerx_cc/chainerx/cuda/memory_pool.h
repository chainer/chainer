#pragma once

#include <functional>
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

// Allocation unit size. This value should be a multiple of the allocation unit size of underlying CUDA allocator.
// TODO(imanishi): The unit sizes of CUDA allocators are not documented. It may be dependent on various factors, such as CUDA architectures,
// CUDA runtime and/or allocation functions. It has been observed that `cudaMallocManaged` has an allocation unit size of 4096 bytes in
// certain environment. We should revisit this number later, or perhaps better to determine it at runtime.
// TODO(imanishi): This number is currently used as both the underlying allocation unit size and the allocation unit size of the memory pool
// (the unit size of splitted chunks). We could separate them and fine-tune to optimize to the typical memory usage.
constexpr size_t kAllocationUnitSize = 512;

// If `kCompactionThreshold` or more consecutive empty free lists were found in free bins, executes `CompactFreebins`.
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
    Allocator() = default;
    virtual ~Allocator() = default;

    Allocator(const Allocator&) = delete;
    Allocator(Allocator&&) = delete;
    Allocator& operator=(const Allocator&) = delete;
    Allocator& operator=(Allocator&&) = delete;

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

namespace cuda_internal {

// A chunk that points to a device memory.
//
// A chunk might be a splitted memory block from a larger allocation.
// The prev/next pointers construct a doubly-linked list of contiguous memories
// sorted by those base addresses.
class Chunk {
public:
    Chunk(void* ptr, size_t offset, size_t bytesize)
        : ptr_{reinterpret_cast<void*>(reinterpret_cast<intptr_t>(ptr) + offset)},  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
          bytesize_{bytesize} {}

    ~Chunk() = default;

    Chunk(const Chunk&) = default;
    Chunk(Chunk&&) = default;
    Chunk& operator=(const Chunk&) = default;
    Chunk& operator=(Chunk&&) = default;

    // Splits this chunk into a chunk with the given bytesize and one with the remaining.
    //
    // Modifies the bytesize and the next pointer of this chunk, and returns the chunk for the remaining.
    std::unique_ptr<Chunk> Split(size_t bytesize);

    // Merges this chunk with the next.
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

using FreeList = std::vector<std::unique_ptr<Chunk>>;  // List of free chunks with the same size.

using FreeBinsMap = std::map<size_t, cuda_internal::FreeList>;

}  // namespace cuda_internal

// Memory pool.
// This class is thread safe.
class MemoryPool {
public:
    explicit MemoryPool(int device_index, std::unique_ptr<Allocator> allocator)
        : device_index_{device_index}, allocator_{std::move(allocator)} {}

    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;

    void FreeUnusedBlocks();

    void* Malloc(size_t bytesize);

    // ChainerxError is thrown if ptr is not an in-use memory pointer.
    void Free(void* ptr);

    void FreeNoExcept(void* ptr) noexcept;

    void SetMallocPreprocessHook(std::function<void(MemoryPool&, size_t)> hook);
    void SetMallocPostprocessHook(std::function<void(MemoryPool&, size_t, void*)> hook);
    void SetFreeHook(std::function<void(MemoryPool&, void*)> hook);

private:
    friend class cuda_internal::MemoryPoolTest;  // for unit-tests

    // Rounds up the memory size to fit memory alignment of memory allocation.
    size_t GetAllocationSize(size_t bytesize) { return ((bytesize + kAllocationUnitSize - 1) / kAllocationUnitSize) * kAllocationUnitSize; }
    void PushIntoFreeList(std::unique_ptr<cuda_internal::Chunk> chunk);
    std::unique_ptr<cuda_internal::Chunk> PopFromFreeList(size_t allocation_size);
    std::unique_ptr<cuda_internal::Chunk> RemoveChunkFromFreeList(cuda_internal::Chunk* chunk);

    // Finds the longest consecutive empty free lists that include the section between `it_start` and `it_end`, and removes them from free
    // bins.
    void CompactFreeBins(cuda_internal::FreeBinsMap::iterator it_start, cuda_internal::FreeBinsMap::iterator it_end);

    int device_index_;
    std::unique_ptr<Allocator> allocator_;
    std::unordered_map<void*, std::unique_ptr<cuda_internal::Chunk>> in_use_;  // ptr => cuda_internal::Chunk
    cuda_internal::FreeBinsMap free_bins_;  // allocation size => cuda_internal::FreeList
    std::mutex in_use_mutex_;
    std::mutex free_bins_mutex_;

    std::function<void(MemoryPool&, size_t)> malloc_preprocess_hook_;
    std::function<void(MemoryPool&, size_t, void*)> malloc_postprocess_hook_;
    std::function<void(MemoryPool&, void*)> free_hook_;
};

}  // namespace cuda
}  // namespace chainerx
