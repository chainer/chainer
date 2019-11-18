#include "chainerx/cuda/memory_pool.h"

#include <map>
#include <memory>
#include <mutex>

#include <gtest/gtest.h>

#include "chainerx/error.h"
#include "chainerx/testing/threading.h"

namespace chainerx {
namespace cuda {
namespace cuda_internal {

class MemoryPoolTest {
public:
    static const FreeBinsMap& GetFreeBins(const MemoryPool& pool) { return pool.free_bins_; }
    static const Allocator* GetAllocator(const MemoryPool& pool) { return pool.allocator_.get(); }
};

}  // namespace cuda_internal

namespace {

void* AddOffset(void* ptr, size_t offset) {
    return reinterpret_cast<void*>(reinterpret_cast<intptr_t>(ptr) + offset);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

using Chunk = cuda_internal::Chunk;

TEST(ChunkTest, Split) {
    size_t mem_bytesize = kAllocationUnitSize * 4;
    std::shared_ptr<void> mem = std::make_unique<uint8_t[]>(mem_bytesize);
    Chunk chunk{mem.get(), 0, mem_bytesize};

    // Split a chunk into two chunks: a chunk with smaller size and one with the remaining.
    std::unique_ptr<Chunk> tail = chunk.Split(kAllocationUnitSize * 2);
    EXPECT_EQ(chunk.ptr(), mem.get());
    EXPECT_EQ(chunk.bytesize(), kAllocationUnitSize * 2);
    EXPECT_EQ(chunk.prev(), nullptr);
    EXPECT_EQ(chunk.next()->ptr(), tail->ptr());
    EXPECT_EQ(tail->ptr(), AddOffset(mem.get(), kAllocationUnitSize * 2));
    EXPECT_EQ(tail->bytesize(), kAllocationUnitSize * 2);
    EXPECT_EQ(tail->prev()->ptr(), chunk.ptr());
    EXPECT_EQ(tail->next(), nullptr);

    // Split the chunk which was already once splitted.
    std::unique_ptr<Chunk> tail_of_head = chunk.Split(kAllocationUnitSize);
    EXPECT_EQ(chunk.ptr(), mem.get());
    EXPECT_EQ(chunk.bytesize(), kAllocationUnitSize);
    EXPECT_EQ(chunk.prev(), nullptr);
    EXPECT_EQ(chunk.next()->ptr(), tail_of_head->ptr());
    EXPECT_EQ(tail_of_head->ptr(), AddOffset(mem.get(), kAllocationUnitSize));
    EXPECT_EQ(tail_of_head->bytesize(), kAllocationUnitSize);
    EXPECT_EQ(tail_of_head->prev()->ptr(), chunk.ptr());
    EXPECT_EQ(tail_of_head->next()->ptr(), tail->ptr());

    // Split the remaining chunk.
    std::unique_ptr<Chunk> tail_of_tail = tail->Split(kAllocationUnitSize);
    EXPECT_EQ(tail->ptr(), AddOffset(chunk.ptr(), kAllocationUnitSize * 2));
    EXPECT_EQ(tail->bytesize(), kAllocationUnitSize);
    EXPECT_EQ(tail->prev()->ptr(), tail_of_head->ptr());
    EXPECT_EQ(tail->next()->ptr(), tail_of_tail->ptr());
    EXPECT_EQ(tail_of_tail->ptr(), AddOffset(mem.get(), kAllocationUnitSize * 3));
    EXPECT_EQ(tail_of_tail->bytesize(), kAllocationUnitSize);
    EXPECT_EQ(tail_of_tail->prev()->ptr(), tail->ptr());
    EXPECT_EQ(tail_of_tail->next(), nullptr);
}

TEST(ChunkTest, MergeWithNext) {
    size_t mem_bytesize = kAllocationUnitSize * 4;
    std::shared_ptr<void> mem = std::make_unique<uint8_t[]>(mem_bytesize);
    Chunk chunk{mem.get(), 0, mem_bytesize};

    void* chunk_ptr = chunk.ptr();
    size_t chunk_bytesize = chunk.bytesize();

    // Split chunk -> [1, 2, 3, 4]
    std::unique_ptr<Chunk> tail = chunk.Split(kAllocationUnitSize * 2);
    std::unique_ptr<Chunk> head = std::make_unique<Chunk>(chunk);
    void* head_ptr = head->ptr();
    size_t head_bytesize = head->bytesize();
    void* tail_ptr = tail->ptr();
    size_t tail_bytesize = tail->bytesize();
    std::unique_ptr<Chunk> tail_next = tail->Split(kAllocationUnitSize);
    std::unique_ptr<Chunk> head_next = head->Split(kAllocationUnitSize);

    // Merge [1] and [2] into [1, 2].
    head->MergeWithNext();
    EXPECT_EQ(head->ptr(), head_ptr);
    EXPECT_EQ(head->bytesize(), head_bytesize);
    EXPECT_EQ(head->prev(), nullptr);
    EXPECT_EQ(head->next()->ptr(), tail_ptr);

    // Merge [3] and [4] into [3, 4].
    tail->MergeWithNext();
    EXPECT_EQ(tail->ptr(), tail_ptr);
    EXPECT_EQ(tail->bytesize(), tail_bytesize);
    EXPECT_EQ(tail->prev()->ptr(), head_ptr);
    EXPECT_EQ(tail->next(), nullptr);

    // Merge [1, 2] and [3, 4] into [1, 2, 3, 4].
    // Merge chunks which were already one merged.
    head->MergeWithNext();
    EXPECT_EQ(head->ptr(), chunk_ptr);
    EXPECT_EQ(head->bytesize(), chunk_bytesize);
    EXPECT_EQ(head->prev(), nullptr);
    EXPECT_EQ(head->next(), nullptr);

    (void)head_next;
    (void)tail_next;
}

// A dummy allocator to test OutOfMemoryError
class FixedCapacityDummyAllocator : public Allocator {
public:
    explicit FixedCapacityDummyAllocator(size_t capacity) : capacity_{capacity} {}

    MallocStatus Malloc(void** ptr, size_t bytesize) override {
        CHAINERX_ASSERT(bytesize > 0);
        uint8_t* mem{};
        {
            std::lock_guard<std::mutex> lock{sizes_mutex_};
            ++malloc_called_;
            if (capacity_ < bytesize) {
                return MallocStatus::kErrorMemoryAllocation;
            }
            mem = new uint8_t[bytesize];
            sizes_[mem] = bytesize;
            capacity_ -= bytesize;
        }
        *ptr = mem;
        return MallocStatus::kSuccess;
    }
    void Free(void* ptr) noexcept override {
        uint8_t* mem = static_cast<uint8_t*>(ptr);
        {
            std::lock_guard<std::mutex> lock{sizes_mutex_};
            auto it = sizes_.find(mem);
            CHAINERX_ASSERT(it != sizes_.end());
            capacity_ += it->second;
            sizes_.erase(it);
            ++free_called_;
        }
        delete[] mem;
    }

    int malloc_called() const { return malloc_called_; }
    int free_called() const { return free_called_; }

private:
    size_t capacity_;
    int malloc_called_{0};
    int free_called_{0};
    std::mutex sizes_mutex_;
    std::map<void*, size_t> sizes_;
};

class MemoryPoolTestForEachAllocator : public ::testing::TestWithParam<std::shared_ptr<MemoryPool>> {};

TEST_P(MemoryPoolTestForEachAllocator, MallocAndFree) {
    MemoryPool& memory_pool = *GetParam();

    // Allocate two distinct memory areas via allocator.
    void* ptr1 = memory_pool.Malloc(1);
    void* ptr2 = memory_pool.Malloc(1);
    EXPECT_NE(ptr1, ptr2);

    // This memory is stored into the free bins.
    memory_pool.Free(ptr2);

    // Fetche a memory area from the free bins.
    void* ptr3 = memory_pool.Malloc(1);
    EXPECT_EQ(ptr2, ptr3);

    memory_pool.Free(ptr3);
    memory_pool.Free(ptr1);
}

TEST_P(MemoryPoolTestForEachAllocator, MallocAllocationUnitSize) {
    MemoryPool& memory_pool = *GetParam();

    // Allocate a memory area via allocator.
    void* ptr1 = memory_pool.Malloc(100);
    memory_pool.Free(ptr1);

    // Allocate a memory area via allocator, because the free bins do not have any consecutive memory area of such bytesize.
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
    const cuda_internal::FreeBinsMap& free_bins = cuda_internal::MemoryPoolTest::GetFreeBins(memory_pool);

    void* ptr1 = memory_pool.Malloc(1);
    memory_pool.Free(ptr1);
    EXPECT_FALSE(free_bins.empty());

    memory_pool.FreeUnusedBlocks();
    EXPECT_TRUE(free_bins.empty());
}

TEST_P(MemoryPoolTestForEachAllocator, MallocSplit) {
    MemoryPool& memory_pool = *GetParam();

    // Allocate a memory area of 4 unit size length in free bins, and store it in free bins.
    void* ptr = memory_pool.Malloc(kAllocationUnitSize * 4);
    memory_pool.Free(ptr);

    // Take the memory area from free bins and split it into two memory areas.
    void* head = memory_pool.Malloc(kAllocationUnitSize * 2);
    void* tail = memory_pool.Malloc(kAllocationUnitSize * 2);
    EXPECT_EQ(ptr, head);
    EXPECT_EQ(AddOffset(ptr, kAllocationUnitSize * 2), tail);
    memory_pool.Free(head);
    memory_pool.Free(tail);
}

TEST_P(MemoryPoolTestForEachAllocator, FreeMerge) {
    MemoryPool& memory_pool = *GetParam();
    void* ptr = memory_pool.Malloc(kAllocationUnitSize * 4);
    memory_pool.Free(ptr);

    // Merge head into tail
    {
        void* head = memory_pool.Malloc(kAllocationUnitSize * 2);
        void* tail = memory_pool.Malloc(kAllocationUnitSize * 2);
        EXPECT_EQ(ptr, head);
        EXPECT_EQ(AddOffset(ptr, kAllocationUnitSize * 2), tail);
        memory_pool.Free(tail);
        memory_pool.Free(head);
        void* p = memory_pool.Malloc(kAllocationUnitSize * 4);
        EXPECT_EQ(ptr, p);
        memory_pool.Free(p);
    }

    // Merge tail into head
    {
        void* head = memory_pool.Malloc(kAllocationUnitSize * 2);
        void* tail = memory_pool.Malloc(kAllocationUnitSize * 2);
        EXPECT_EQ(ptr, head);
        EXPECT_EQ(AddOffset(ptr, kAllocationUnitSize * 2), tail);
        memory_pool.Free(head);
        memory_pool.Free(tail);
        void* p = memory_pool.Malloc(kAllocationUnitSize * 4);
        EXPECT_EQ(ptr, p);
        memory_pool.Free(p);
    }
}

TEST_P(MemoryPoolTestForEachAllocator, MallocSizeIncreasing) {
    static constexpr size_t size1 = kAllocationUnitSize * 4;
    static constexpr size_t size2 = kAllocationUnitSize * 8;

    MemoryPool& memory_pool = *GetParam();
    void* ptr1 = memory_pool.Malloc(size1);
    memory_pool.Free(ptr1);

    // Cannot take the memory area from free bins, because there is no memory area larger or equal to size2.
    void* ptr2 = memory_pool.Malloc(size2);
    memory_pool.Free(ptr2);
    EXPECT_NE(ptr1, ptr2);
}

TEST_P(MemoryPoolTestForEachAllocator, MallocSizeDecreasing) {
    static constexpr size_t size1 = kAllocationUnitSize * 8;
    static constexpr size_t size2 = kAllocationUnitSize * 4;

    MemoryPool& memory_pool = *GetParam();
    void* ptr1 = memory_pool.Malloc(size1);
    memory_pool.Free(ptr1);

    // Take the memory area from free bins, because there is a memory area larger or equal to size2.
    void* ptr2 = memory_pool.Malloc(size2);
    memory_pool.Free(ptr2);
    EXPECT_EQ(ptr1, ptr2);
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

    size_t size1 = 1U;
    size_t size2 = kCapacity;

    void* ptr1 = memory_pool.Malloc(size1);  // no throw
    memory_pool.Free(ptr1);

    // There is no memory area larger or equal to size2, so the memory pool tries to fetch an new memory area from the allocator. However,
    // there is no free space in the allocator, so the memory pool frees all unused blocks and tries to fetch an new memory area again.
    // Finally, the memory pool succeeds to obtain an memory area and returns it.
    void* ptr2 = memory_pool.Malloc(size2);  // no throw
    memory_pool.Free(ptr2);

    EXPECT_EQ(allocator->malloc_called(), 3);
}

TEST(MemoryPoolTest, FreeUnusedBlocksSplitAndFreeTail) {
    // Do not free splitted blocks
    static constexpr size_t kCapacity = cuda::kAllocationUnitSize * 4;
    MemoryPool memory_pool{0, std::make_unique<FixedCapacityDummyAllocator>(kCapacity)};
    auto allocator = dynamic_cast<const FixedCapacityDummyAllocator*>(cuda_internal::MemoryPoolTest::GetAllocator(memory_pool));

    void* ptr = memory_pool.Malloc(kAllocationUnitSize * 4);
    memory_pool.Free(ptr);

    memory_pool.Malloc(kAllocationUnitSize * 2);
    void* tail = memory_pool.Malloc(kAllocationUnitSize * 2);
    memory_pool.Free(tail);

    // The memory pool has an unused block in free bins, but it should not be freed because the previous memory area is in use.
    memory_pool.FreeUnusedBlocks();
    void* ptr2 = memory_pool.Malloc(kAllocationUnitSize * 2);
    EXPECT_EQ(tail, ptr2);
    EXPECT_EQ(allocator->free_called(), 0);
}

TEST(MemoryPoolTest, FreeUnusedBlocksSplitAndFreeHead) {
    // Do not free splitted blocks
    static constexpr size_t kCapacity = cuda::kAllocationUnitSize * 4;
    MemoryPool memory_pool{0, std::make_unique<FixedCapacityDummyAllocator>(kCapacity)};
    auto allocator = dynamic_cast<const FixedCapacityDummyAllocator*>(cuda_internal::MemoryPoolTest::GetAllocator(memory_pool));

    void* ptr = memory_pool.Malloc(kAllocationUnitSize * 4);
    memory_pool.Free(ptr);

    void* head = memory_pool.Malloc(kAllocationUnitSize * 2);
    memory_pool.Malloc(kAllocationUnitSize * 2);

    // The memory pool has an unused block in free bins, but it should not be freed because the next memory area is in use.
    memory_pool.Free(head);
    memory_pool.FreeUnusedBlocks();
    void* ptr2 = memory_pool.Malloc(kAllocationUnitSize * 2);
    EXPECT_EQ(head, ptr2);
    EXPECT_EQ(allocator->free_called(), 0);
}

TEST(MemoryPoolTest, FreeUnusedBlocksThreadSafe) {
    static constexpr size_t kRepeat = 100U;
    MemoryPool memory_pool{0, std::make_unique<FixedCapacityDummyAllocator>(0xffffffffU)};

    // TODO(niboshi): Use TEST_THREAD_SAFE. Currently it depends on thread sanitizer and does not work with CUDA.
    testing::RunThreads(2, [&memory_pool](size_t thread_index) {
        for (size_t i = 0; i < kRepeat; ++i) {
            switch (thread_index) {
                case 0: {
                    void* ptr = memory_pool.Malloc(1U);
                    memory_pool.Free(ptr);
                    break;
                }
                case 1:
                    memory_pool.FreeUnusedBlocks();
                    break;
                default:
                    CHAINERX_NEVER_REACH();
            }
        }
    });
}

TEST(MemoryPoolTest, MallocFreeThreadSafe) {
    static constexpr size_t kRepeat = 100U;
    MemoryPool memory_pool{0, std::make_unique<FixedCapacityDummyAllocator>(0xffffffffU)};

    // TODO(niboshi): Use TEST_THREAD_SAFE. Currently it depends on thread sanitizer and does not work with CUDA.
    testing::RunThreads(2, [&memory_pool](size_t thread_index) {
        for (size_t i = 0; i < kRepeat; ++i) {
            switch (thread_index) {
                case 0: {
                    void* ptr = memory_pool.Malloc(1U);
                    memory_pool.Free(ptr);
                    break;
                }
                case 1: {
                    void* ptr = memory_pool.Malloc(1U);
                    memory_pool.Free(ptr);
                    break;
                }
            }
        }
    });
}

TEST(MemoryPoolTest, Hook) {
    MemoryPool memory_pool{0, std::make_unique<FixedCapacityDummyAllocator>(0xffffffffU)};

    size_t total_memory = 0;
    std::map<void*, size_t> memories;
    auto malloc_postprocess_hook = [&total_memory, &memories](MemoryPool&, size_t bytesize, void* ptr) {
        memories[ptr] += bytesize;
        total_memory += bytesize;
    };
    auto free_preprocess_hook = [&total_memory, &memories](MemoryPool&, void* ptr) {
        auto found = memories.find(ptr);
        ASSERT_TRUE(found != memories.end());
        const size_t bytesize = found->second;
        memories.erase(found);
        total_memory -= bytesize;
    };
    memory_pool.SetMallocPostprocessHook(malloc_postprocess_hook);
    memory_pool.SetFreeHook(free_preprocess_hook);

    EXPECT_EQ(total_memory, 0U);
    EXPECT_EQ(memories.size(), 0U);

    void* ptr1 = memory_pool.Malloc(12);

    EXPECT_EQ(total_memory, 12U);
    EXPECT_EQ(memories.size(), 1U);

    void* ptr2 = memory_pool.Malloc(300);

    EXPECT_EQ(total_memory, 312U);
    EXPECT_EQ(memories.size(), 2U);

    memory_pool.Free(ptr1);

    EXPECT_EQ(total_memory, 300U);
    EXPECT_EQ(memories.size(), 1U);

    memory_pool.Free(ptr2);

    EXPECT_EQ(total_memory, 0U);
    EXPECT_EQ(memories.size(), 0U);
}

}  // namespace
}  // namespace cuda
}  // namespace chainerx
