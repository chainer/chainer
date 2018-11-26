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
    // TODO(sonots): Reconsider ptr type, and reinterpret_cast
    Chunk(const std::shared_ptr<void>& mem, size_t offset, size_t size)
        : mem_(mem), ptr_(reinterpret_cast<intptr_t>(mem.get()) + offset), offset_(offset), size_(size) {}
    Chunk(const Chunk&) = default;
    ~Chunk() {}

    // Split contiguous block of a larger allocation
    friend std::shared_ptr<Chunk> Split(std::shared_ptr<Chunk>& self, size_t size);

    // Merge previously splitted block (chunk)
    friend void Merge(std::shared_ptr<Chunk>& self, std::shared_ptr<Chunk> remaining);

    void* ptr() const { return reinterpret_cast<void*>(ptr_); }
    size_t offset() const { return offset_; }
    size_t size() const { return size_; }
    const std::shared_ptr<Chunk>& prev() const { return prev_; }
    std::shared_ptr<Chunk>& prev() { return prev_; }
    const std::shared_ptr<Chunk>& next() const { return next_; }
    std::shared_ptr<Chunk>& next() { return next_; }
    bool in_use() const { return in_use_; }

    void SetPrev(const std::shared_ptr<Chunk>& prev) { prev_ = prev; }
    void SetNext(const std::shared_ptr<Chunk>& next) { next_ = next; }
    void SetInUse(bool in_use) { in_use_ = in_use; }

private:
    std::shared_ptr<void> mem_;  // The memory buffer.
    intptr_t ptr_ = 0;  // Memory address.
    size_t offset_ = 0;  // An offset bytes from the head of the buffer.
    size_t size_ = 0;  // Chunk size in bytes.
    int device_index_;  // GPU device id whose memory the pointer refers to.
    std::shared_ptr<Chunk> prev_;  // prev memory pointer if split from a larger allocation
    std::shared_ptr<Chunk> next_;  // next memory pointer if split from a larger allocation
    bool in_use_ = false;  // chunk is in use
};

// Memory pool.
// This class is thread safe.
class MemoryPool {
public:
    explicit MemoryPool(int device_index, std::unique_ptr<Allocator> allocator)
        : device_index_{device_index}, allocator_{std::move(allocator)} {}

    ~MemoryPool();

    void FreeUnusedBlocks();

    void* Malloc(size_t bytesize);

    // ChainerxError is thrown if ptr is not an in-use memory pointer.
    void Free(void* ptr);

    void FreeNoExcept(void* ptr) noexcept;

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
