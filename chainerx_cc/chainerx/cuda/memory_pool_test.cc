#include "chainerx/cuda/memory_pool.h"

#include <memory>

#include <gtest/gtest.h>

#include "chainerx/error.h"

namespace chainerx {
namespace cuda {
namespace cuda_internal {

class MemoryPoolTest {
public:
    static const std::vector<std::vector<void*>>& GetFreeBins(const MemoryPool& pool) { return pool.free_bins_; }
    static const Allocator* GetAllocator(const MemoryPool& pool) { return pool.allocator_.get(); }
};

}  // namespace cuda_internal

namespace {

void* AddOffset(void* ptr, size_t offset) { return reinterpret_cast<void*>(reinterpret_cast<intptr_t>(ptr) + offset); }

TEST(ChunkTest, Split) {
    size_t mem_bytesize = kAllocationUnitSize * 4;
    std::shared_ptr<void> mem = std::make_unique<uint8_t[]>(mem_bytesize);
    Chunk chunk{mem, 0, mem_bytesize};

    std::unique_ptr<Chunk> tail = chunk.Split(kAllocationUnitSize * 2);
    EXPECT_EQ(chunk.ptr(), mem.get());
    EXPECT_EQ(chunk.offset(), size_t{0});
    EXPECT_EQ(chunk.bytesize(), kAllocationUnitSize * 2);
    EXPECT_EQ(chunk.prev(), nullptr);
    EXPECT_EQ(chunk.next()->ptr(), tail->ptr());
    EXPECT_EQ(tail->ptr(), AddOffset(mem.get(), kAllocationUnitSize * 2));
    EXPECT_EQ(tail->offset(), kAllocationUnitSize * 2);
    EXPECT_EQ(tail->bytesize(), kAllocationUnitSize * 2);
    EXPECT_EQ(tail->prev()->ptr(), chunk.ptr());
    EXPECT_EQ(tail->next(), nullptr);

    std::unique_ptr<Chunk> tail_of_head = chunk.Split(kAllocationUnitSize);
    EXPECT_EQ(chunk.ptr(), mem.get());
    EXPECT_EQ(chunk.offset(), size_t{0});
    EXPECT_EQ(chunk.bytesize(), kAllocationUnitSize);
    EXPECT_EQ(chunk.prev(), nullptr);
    EXPECT_EQ(chunk.next()->ptr(), tail_of_head->ptr());
    EXPECT_EQ(tail_of_head->ptr(), AddOffset(mem.get(), kAllocationUnitSize));
    EXPECT_EQ(tail_of_head->offset(), kAllocationUnitSize);
    EXPECT_EQ(tail_of_head->bytesize(), kAllocationUnitSize);
    EXPECT_EQ(tail_of_head->prev()->ptr(), chunk.ptr());
    EXPECT_EQ(tail_of_head->next()->ptr(), tail->ptr());

    std::unique_ptr<Chunk> tail_of_tail = tail->Split(kAllocationUnitSize);
    EXPECT_EQ(tail->ptr(), AddOffset(chunk.ptr(), kAllocationUnitSize * 2));
    EXPECT_EQ(tail->offset(), kAllocationUnitSize * 2);
    EXPECT_EQ(tail->bytesize(), kAllocationUnitSize);
    EXPECT_EQ(tail->prev()->ptr(), tail_of_head->ptr());
    EXPECT_EQ(tail->next()->ptr(), tail_of_tail->ptr());
    EXPECT_EQ(tail_of_tail->ptr(), AddOffset(mem.get(), kAllocationUnitSize * 3));
    EXPECT_EQ(tail_of_tail->offset(), kAllocationUnitSize * 3);
    EXPECT_EQ(tail_of_tail->bytesize(), kAllocationUnitSize);
    EXPECT_EQ(tail_of_tail->prev()->ptr(), tail->ptr());
    EXPECT_EQ(tail_of_tail->next(), nullptr);
}

TEST(ChunkTest, MergeWithNext) {
    size_t mem_bytesize = kAllocationUnitSize * 4;
    std::shared_ptr<void> mem = std::make_unique<uint8_t[]>(mem_bytesize);
    Chunk chunk{mem, 0, mem_bytesize};

    void* chunk_ptr = chunk.ptr();
    size_t chunk_offset = chunk.offset();
    size_t chunk_bytesize = chunk.bytesize();

    std::unique_ptr<Chunk> tail = chunk.Split(kAllocationUnitSize * 2);
    std::unique_ptr<Chunk> head = std::make_unique<Chunk>(chunk);
    void* head_ptr = head->ptr();
    size_t head_offset = head->offset();
    size_t head_bytesize = head->bytesize();
    void* tail_ptr = tail->ptr();
    size_t tail_offset = tail->offset();
    size_t tail_bytesize = tail->bytesize();

    head->Split(kAllocationUnitSize);
    tail->Split(kAllocationUnitSize);

    head->MergeWithNext();
    EXPECT_EQ(head->ptr(), head_ptr);
    EXPECT_EQ(head->offset(), head_offset);
    EXPECT_EQ(head->bytesize(), head_bytesize);
    EXPECT_EQ(head->prev(), nullptr);
    EXPECT_EQ(head->next()->ptr(), tail_ptr);

    tail->MergeWithNext();
    EXPECT_EQ(tail->ptr(), tail_ptr);
    EXPECT_EQ(tail->offset(), tail_offset);
    EXPECT_EQ(tail->bytesize(), tail_bytesize);
    EXPECT_EQ(tail->prev()->ptr(), head_ptr);
    EXPECT_EQ(tail->next(), nullptr);

    head->MergeWithNext();
    EXPECT_EQ(head->ptr(), chunk_ptr);
    EXPECT_EQ(head->offset(), chunk_offset);
    EXPECT_EQ(head->bytesize(), chunk_bytesize);
    EXPECT_EQ(head->prev(), nullptr);
    EXPECT_EQ(head->next(), nullptr);
}

// A dummy allocator to test OutOfMemoryError
class FixedCapacityDummyAllocator : public Allocator {
public:
    explicit FixedCapacityDummyAllocator(size_t capacity) : capacity_{capacity} {}

    MallocStatus Malloc(void** ptr, size_t bytesize) override {
        CHAINERX_ASSERT(bytesize > 0);
        ++malloc_called_;
        if (capacity_ < bytesize) {
            return MallocStatus::kErrorMemoryAllocation;
        }
        // bytesize is encoded in the dummy pointer.
        auto i = static_cast<intptr_t>(bytesize);
        *ptr = reinterpret_cast<void*>(i);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        capacity_ -= bytesize;
        return MallocStatus::kSuccess;
    }
    void Free(void* ptr) noexcept override {
        intptr_t i = reinterpret_cast<intptr_t>(ptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        capacity_ += static_cast<size_t>(i);
    }

    int malloc_called() const { return malloc_called_; }

private:
    size_t capacity_;
    int malloc_called_{0};
};

class MemoryPoolTestForEachAllocator : public ::testing::TestWithParam<std::shared_ptr<MemoryPool>> {};

TEST_P(MemoryPoolTestForEachAllocator, MallocAndFree) {
    MemoryPool& memory_pool = *GetParam();

    void* ptr1 = memory_pool.Malloc(1);
    void* ptr2 = memory_pool.Malloc(1);
    EXPECT_NE(ptr1, ptr2);

    memory_pool.Free(ptr2);

    void* ptr3 = memory_pool.Malloc(1);
    EXPECT_EQ(ptr2, ptr3);
    memory_pool.Free(ptr3);

    memory_pool.Free(ptr1);
}

TEST_P(MemoryPoolTestForEachAllocator, MallocAllocationUnitSize) {
    MemoryPool& memory_pool = *GetParam();

    void* ptr1 = memory_pool.Malloc(100);
    memory_pool.Free(ptr1);

    void* ptr2 = memory_pool.Malloc(100 + kAllocationUnitSize);
    EXPECT_NE(ptr1, ptr2);

    memory_pool.Free(ptr2);
}

TEST_P(MemoryPoolTestForEachAllocator, MallocZeroByte) {
    MemoryPool& memory_pool = *GetParam();
    void* ptr = memory_pool.Malloc(0);
    EXPECT_EQ(nullptr, ptr);
    memory_pool.Free(ptr);  // no throw
}

TEST_P(MemoryPoolTestForEachAllocator, FreeTwice) {
    MemoryPool& memory_pool = *GetParam();
    void* ptr = memory_pool.Malloc(1);
    memory_pool.Free(ptr);
    EXPECT_THROW(memory_pool.Free(ptr), ChainerxError);
}

TEST_P(MemoryPoolTestForEachAllocator, FreeForeignPointer) {
    MemoryPool& memory_pool = *GetParam();
    void* ptr = &memory_pool;
    EXPECT_THROW(memory_pool.Free(ptr), ChainerxError);
}

TEST_P(MemoryPoolTestForEachAllocator, FreeUnusedBlocks) {
    MemoryPool& memory_pool = *GetParam();
    const std::vector<std::vector<void*>>& free_bins = cuda_internal::MemoryPoolTest::GetFreeBins(memory_pool);

    void* ptr1 = memory_pool.Malloc(1);
    memory_pool.Free(ptr1);
    EXPECT_FALSE(free_bins.empty());

    memory_pool.FreeUnusedBlocks();
    EXPECT_TRUE(free_bins.empty());
}

INSTANTIATE_TEST_CASE_P(
        ForEachAllocator,
        MemoryPoolTestForEachAllocator,
        ::testing::Values(
                std::make_shared<MemoryPool>(0, std::make_unique<DeviceMemoryAllocator>()),
                std::make_shared<MemoryPool>(0, std::make_unique<PinnedMemoryAllocator>())));

TEST(MemoryPoolTest, MallocThrowOutOfMemory) {
    MemoryPool memory_pool{0, std::make_unique<FixedCapacityDummyAllocator>(0U)};
    EXPECT_THROW(memory_pool.Malloc(1), OutOfMemoryError);
}

TEST(MemoryPoolTest, MallocRetryOutOfMemory) {
    static constexpr size_t kCapacity = cuda::kAllocationUnitSize * 4;
    MemoryPool memory_pool{0, std::make_unique<FixedCapacityDummyAllocator>(kCapacity)};
    auto allocator = dynamic_cast<const FixedCapacityDummyAllocator*>(cuda_internal::MemoryPoolTest::GetAllocator(memory_pool));

    size_t size1 = kCapacity;
    size_t size2 = 1U;

    void* ptr1 = memory_pool.Malloc(size1);  // no throw
    memory_pool.Free(ptr1);

    void* ptr2 = memory_pool.Malloc(size2);  // no throw
    memory_pool.Free(ptr2);

    EXPECT_EQ(allocator->malloc_called(), 3);
}

}  // namespace
}  // namespace cuda
}  // namespace chainerx
