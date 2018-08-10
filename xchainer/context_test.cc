#include "xchainer/context.h"

#include <cstdlib>
#include <future>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/backend.h"
#include "xchainer/device.h"
#include "xchainer/native/native_backend.h"
#include "xchainer/native/native_device.h"
#include "xchainer/testing/threading.h"

namespace xchainer {
namespace {

TEST(ContextTest, Ctor) {
    Context();  // no throw
}

TEST(ContextTest, GetBackend) {
    Context ctx;
    Backend& backend = ctx.GetBackend("native");
    EXPECT_EQ(&backend, &ctx.GetBackend("native"));
}

TEST(ContextTest, NativeBackend) {
    Context ctx;
    native::NativeBackend& backend = ctx.GetNativeBackend();
    EXPECT_EQ(&ctx.GetBackend("native"), &backend);
}

TEST(ContextTest, GetBackendThreadSafe) {
    static constexpr size_t kRepeat = 100;
    static constexpr size_t kThreadCount = 128;
    std::string backend_name{"native"};

    xchainer::testing::CheckThreadSafety(
            kRepeat,
            kThreadCount,
            [](size_t /*repeat*/) { return std::make_unique<Context>(); },
            [&backend_name](size_t /*thread_index*/, const std::unique_ptr<Context>& ctx) {
                Backend& backend = ctx->GetBackend(backend_name);
                return &backend;
            },
            [&backend_name](const std::vector<Backend*>& results) {
                for (Backend* backend : results) {
                    ASSERT_EQ(backend, results.front());
                    ASSERT_EQ(backend_name, backend->GetName());
                }
            });
}

TEST(ContextTest, BackendNotFound) {
    Context ctx;
    EXPECT_THROW(ctx.GetBackend("something_that_does_not_exist"), BackendError);
}

TEST(ContextTest, GetDevice) {
    Context ctx;
    Device& device = ctx.GetDevice({"native", 0});
    EXPECT_EQ(&device, &ctx.GetDevice({"native:0"}));
}

TEST(ContextTest, GetDeviceThreadSafe) {
    static constexpr size_t kRepeat = 100;
    static constexpr int kDeviceCount = 4;
    static constexpr size_t kThreadCountPerDevice = 32;
    static constexpr size_t kThreadCount = kDeviceCount * kThreadCountPerDevice;

    xchainer::testing::CheckThreadSafety(
            kRepeat,
            kThreadCount,
            [](size_t /*repeat*/) { return std::make_unique<Context>(); },
            [](size_t thread_index, const std::unique_ptr<Context>& ctx) {
                int device_index = thread_index / kThreadCountPerDevice;
                Device& device = ctx->GetDevice({"native", device_index});
                return &device;
            },
            [](const std::vector<Device*>& results) {
                // Check device pointers are identical within each set of threads corresponding to one device
                for (int device_index = 0; device_index < kDeviceCount; ++device_index) {
                    auto it_first = std::next(results.begin(), device_index * kThreadCountPerDevice);
                    auto it_last = std::next(results.begin(), (device_index + 1) * kThreadCountPerDevice);
                    Device* ref_device = *it_first;

                    // Check the device index
                    ASSERT_EQ(device_index, ref_device->index());

                    for (auto it = it_first; it != it_last; ++it) {
                        ASSERT_EQ(ref_device, *it);
                    }
                }
            });
}

TEST(ContextTest, DefaultContext) {
    SetGlobalDefaultContext(nullptr);
    SetDefaultContext(nullptr);
    ASSERT_THROW(GetDefaultContext(), XchainerError);

    Context ctx;
    SetGlobalDefaultContext(nullptr);
    SetDefaultContext(&ctx);
    ASSERT_EQ(&ctx, &GetDefaultContext());

    Context global_ctx;
    SetGlobalDefaultContext(&global_ctx);
    SetDefaultContext(nullptr);
    ASSERT_EQ(&global_ctx, &GetDefaultContext());

    SetGlobalDefaultContext(&global_ctx);
    SetDefaultContext(&ctx);
    ASSERT_EQ(&ctx, &GetDefaultContext());
}

TEST(ContextTest, DefaultContextThreadSafe) {
    static constexpr size_t kRepeat = 100;
    static constexpr size_t kThreadCount = 256;

    xchainer::testing::CheckThreadSafety(
            kRepeat,
            kThreadCount,
            [](size_t /*repeat*/) { return nullptr; },
            [](size_t /*thread_index*/, std::nullptr_t) {
                Context ctx{};
                SetDefaultContext(&ctx);
                Context& ctx2 = GetDefaultContext();
                EXPECT_EQ(&ctx, &ctx2);
                return nullptr;
            },
            [](const std::vector<std::nullptr_t>& /*results*/) {});
}

TEST(ContextTest, GlobalDefaultContextThreadSafe) {
    static constexpr size_t kRepeat = 100;
    static constexpr size_t kThreadCount = 256;

    // Each of SetGlobalDefaultContext() and GetGlobalDefaultContext() must be thread-safe, but a pair of these calls is not guaranteed to
    // be so. In this check, a single context is set as the global context simultaneously in many threads and it only checks that
    // the succeeding Get...() call returns the same instance.
    xchainer::testing::CheckThreadSafety(
            kRepeat,
            kThreadCount,
            [](size_t /*repeat*/) { return std::make_unique<Context>(); },
            [](size_t /*thread_index*/, const std::unique_ptr<Context>& ctx) {
                SetGlobalDefaultContext(ctx.get());
                Context& ctx2 = GetGlobalDefaultContext();
                EXPECT_EQ(ctx.get(), &ctx2);
                return nullptr;
            },
            [](const std::vector<std::nullptr_t>& /*results*/) {});
}

TEST(ContextTest, GlobalDefaultContext) {
    SetGlobalDefaultContext(nullptr);
    ASSERT_THROW(GetGlobalDefaultContext(), XchainerError);

    Context ctx;
    SetGlobalDefaultContext(&ctx);
    ASSERT_EQ(&ctx, &GetGlobalDefaultContext());
}

TEST(ContextTest, ThreadLocal) {
    Context ctx;
    SetDefaultContext(&ctx);

    Context ctx2;
    auto future = std::async(std::launch::async, [&ctx2] {
        SetDefaultContext(&ctx2);
        return &GetDefaultContext();
    });
    ASSERT_NE(&GetDefaultContext(), future.get());
}

TEST(ContextTest, ContextScopeCtor) {
    SetGlobalDefaultContext(nullptr);
    SetDefaultContext(nullptr);
    Context ctx1;
    {
        // ContextScope should work even if default context is not set
        ContextScope scope(ctx1);
        EXPECT_EQ(&ctx1, &GetDefaultContext());
    }
    ASSERT_THROW(GetDefaultContext(), XchainerError);
    SetDefaultContext(&ctx1);
    {
        Context ctx2;
        ContextScope scope(ctx2);
        EXPECT_EQ(&ctx2, &GetDefaultContext());
    }
    ASSERT_EQ(&ctx1, &GetDefaultContext());
    {
        ContextScope scope;
        EXPECT_EQ(&ctx1, &GetDefaultContext());
        Context ctx2;
        SetDefaultContext(&ctx2);
    }
    ASSERT_EQ(&ctx1, &GetDefaultContext());
    Context ctx2;
    {
        ContextScope scope(ctx2);
        scope.Exit();
        EXPECT_EQ(&ctx1, &GetDefaultContext());
        SetDefaultContext(&ctx2);
        // not recovered here because the scope has already existed
    }
    ASSERT_EQ(&ctx2, &GetDefaultContext());
}

TEST(ContextTest, ContextScopeResetDevice) {
    SetGlobalDefaultContext(nullptr);
    SetDefaultContext(nullptr);
    Context ctx1;
    Context ctx2;
    {
        ContextScope ctx_scope1{ctx1};
        Device& device1 = ctx1.GetDevice({"native", 0});
        DeviceScope dev_scope1{device1};

        {
            ContextScope ctx_scope2{ctx2};
            ASSERT_NE(&device1, &GetDefaultDevice());
            Device& device2 = ctx2.GetDevice({"native", 0});
            SetDefaultDevice(&device2);
        }

        EXPECT_EQ(&device1, &GetDefaultDevice());
    }
}

TEST(ContextTest, UserDefinedBackend) {
    ::setenv("XCHAINER_PATH", XCHAINER_TEST_DIR "/context_testdata", 1);
    Context ctx;
    Backend& backend0 = ctx.GetBackend("backend0");
    EXPECT_EQ("backend0", backend0.GetName());
    Backend& backend0_2 = ctx.GetBackend("backend0");
    EXPECT_EQ(&backend0, &backend0_2);
    Backend& backend1 = ctx.GetBackend("backend1");
    EXPECT_EQ("backend1", backend1.GetName());

    Device& device0 = ctx.GetDevice(std::string("backend0:0"));
    EXPECT_EQ(&backend0, &device0.backend());
}

TEST(ContextTest, GetBackendOnDefaultContext) {
    // xchainer::GetBackend
    Context ctx;
    SetDefaultContext(&ctx);
    Backend& backend = GetBackend("native");
    EXPECT_EQ(&ctx, &backend.context());
    EXPECT_EQ("native", backend.GetName());
}

TEST(ContextTest, GetNativeBackendOnDefaultContext) {
    // xchainer::GetNativeBackend
    Context ctx;
    SetDefaultContext(&ctx);
    native::NativeBackend& backend = GetNativeBackend();
    EXPECT_EQ(&ctx.GetNativeBackend(), &backend);
}

TEST(ContextTest, GetDeviceOnDefaultContext) {
    // xchainer::GetDevice
    Context ctx;
    SetDefaultContext(&ctx);
    Device& device = GetDevice({"native:0"});
    EXPECT_EQ(&ctx, &device.backend().context());
    EXPECT_EQ("native:0", device.name());
}

}  // namespace
}  // namespace xchainer
