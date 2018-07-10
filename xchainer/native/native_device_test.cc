#include "xchainer/native/native_device.h"

#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "xchainer/context.h"
#include "xchainer/native/native_backend.h"

namespace xchainer {
namespace native {
namespace {

template <typename T>
void ExpectDataEqual(const std::shared_ptr<void>& expected, const std::shared_ptr<void>& actual, size_t size) {
    auto expected_raw_ptr = static_cast<const T*>(expected.get());
    auto actual_raw_ptr = static_cast<const T*>(actual.get());
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(expected_raw_ptr[i], actual_raw_ptr[i]);
    }
}

NativeDevice& GetNativeDevice(Context& ctx, int device_index) {
    // Using dynamic_cast to ensure it's actually NativeDevice
    return dynamic_cast<NativeDevice&>(ctx.GetDevice({"native", device_index}));
}

TEST(NativeDeviceTest, Allocate) {
    Context ctx;
    NativeDevice& device = GetNativeDevice(ctx, 0);

    size_t bytesize = 3;
    std::shared_ptr<void> ptr = device.Allocate(bytesize);
    EXPECT_NE(nullptr, ptr);
}

TEST(NativeDeviceTest, AllocateZero) {
    Context ctx;
    NativeDevice& device = GetNativeDevice(ctx, 0);

    std::shared_ptr<void> ptr = device.Allocate(size_t{0});
    EXPECT_NE(ptr, nullptr);
}

TEST(NativeDeviceTest, AllocateFreeThreadSafe) {
    static constexpr size_t kNumThreads = 1024;
    static constexpr size_t kNumLoopsPerThread = 128;
    Context ctx;
    NativeDevice& device = GetNativeDevice(ctx, 0);

    // Allocate-and-free loop
    auto func = [&device](size_t size) {
        for (size_t j = 0; j < kNumLoopsPerThread; ++j) {
            std::shared_ptr<void> ptr = device.Allocate(size);
            (void)ptr;  // unused
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (size_t i = 0; i < kNumThreads; ++i) {
        threads.emplace_back(func, i);
    }

    // Join threads
    for (size_t i = 0; i < kNumThreads; ++i) {
        threads[i].join();
    }
}

TEST(NativeDeviceTest, MakeDataFromForeignPointer) {
    Context ctx;
    NativeDevice& device = GetNativeDevice(ctx, 0);

    size_t bytesize = 3;
    std::shared_ptr<void> data = device.Allocate(bytesize);
    EXPECT_EQ(data.get(), device.MakeDataFromForeignPointer(data).get());
}

TEST(NativeDeviceTest, FromHostMemory) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src(raw_data, [](const float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    NativeDevice& device = GetNativeDevice(ctx, 0);

    std::shared_ptr<void> dst = device.FromHostMemory(src, bytesize);
    EXPECT_EQ(src.get(), dst.get());
}

TEST(NativeDeviceTest, Synchronize) {
    Context ctx;
    NativeDevice& device = GetNativeDevice(ctx, 0);
    device.Synchronize();  // no throw
}

}  // namespace
}  // namespace native
}  // namespace xchainer
