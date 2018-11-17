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

// A dummy allocator to test throw OutOfMemoryError
class AlwaysOutOfMemoryAllocator : public Allocator {
public:
    MallocStatus Malloc(void** ptr, size_t bytesize) override {
        (void)ptr;  // unused
        (void)bytesize;  // unused
        return MallocStatus::kErrorMemoryAllocation;
    }
    void Free(void* ptr) noexcept override {
        (void)ptr;  // unused
    }
};

// A dummy allocator to test retry on out of memory
class OnceOutOfMemoryAllocator : public Allocator {
public:
    MallocStatus Malloc(void** ptr, size_t bytesize) override {
        (void)ptr;  // unused
        (void)bytesize;  // unused
        if (malloc_called_++ == 0) {
            return MallocStatus::kErrorMemoryAllocation;
        }
        return MallocStatus::kSuccess;
    }
    void Free(void* ptr) noexcept override {
        (void)ptr;  // unused
    }
    int malloc_called() const { return malloc_called_; }

private:
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
    MemoryPool memory_pool{0, std::make_unique<AlwaysOutOfMemoryAllocator>()};
    EXPECT_THROW(memory_pool.Malloc(1), OutOfMemoryError);
}

TEST(MemoryPoolTest, MallocRetryOutOfMemory) {
    MemoryPool memory_pool{0, std::make_unique<OnceOutOfMemoryAllocator>()};
    void* ptr = memory_pool.Malloc(1);  // no throw
    auto allocator = dynamic_cast<const OnceOutOfMemoryAllocator*>(cuda_internal::MemoryPoolTest::GetAllocator(memory_pool));
    EXPECT_EQ(allocator->malloc_called(), 2);
    memory_pool.Free(ptr);
}

}  // namespace
}  // namespace cuda
}  // namespace chainerx
