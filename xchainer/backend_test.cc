#include "xchainer/backend.h"

#include <future>

#include <gtest/gtest.h>

#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/error.h"
#include "xchainer/native_backend.h"

namespace xchainer {
namespace {

TEST(BackendTest, SetCurrentBackend) {
    ASSERT_THROW(GetCurrentBackend(), XchainerError);

    auto native_backend = std::make_unique<NativeBackend>();
    SetCurrentBackend(native_backend.get());
    ASSERT_EQ(native_backend.get(), GetCurrentBackend());

#ifdef XCHAINER_ENABLE_CUDA
    auto cuda_backend = std::make_unique<cuda::CudaBackend>();
    SetCurrentBackend(cuda_backend.get());
    ASSERT_EQ(cuda_backend.get(), GetCurrentBackend());
#endif  // XCHAINER_ENABLE_CUDA

    auto native_backend2 = std::make_unique<NativeBackend>();
    SetCurrentBackend(native_backend2.get());
    ASSERT_EQ(native_backend2.get(), GetCurrentBackend());
}

TEST(BackendTest, ThreadLocal) {
    auto backend1 = std::make_unique<NativeBackend>();
    SetCurrentBackend(backend1.get());

    auto future = std::async(std::launch::async, [] {
        auto backend2 = std::make_unique<NativeBackend>();
        SetCurrentBackend(backend2.get());
        return GetCurrentBackend();
    });
    ASSERT_NE(GetCurrentBackend(), future.get());
}

TEST(BackendScopeTest, Ctor) {
    auto backend1 = std::make_unique<NativeBackend>();
    SetCurrentBackend(backend1.get());
    {
        auto backend2 = std::make_unique<NativeBackend>();
        BackendScope scope(backend2.get());
        EXPECT_EQ(backend2.get(), GetCurrentBackend());
    }
    ASSERT_EQ(backend1.get(), GetCurrentBackend());
    {
        BackendScope scope;
        EXPECT_EQ(backend1.get(), GetCurrentBackend());
        auto backend2 = std::make_unique<NativeBackend>();
        SetCurrentBackend(backend2.get());
    }
    ASSERT_EQ(backend1.get(), GetCurrentBackend());
    auto backend2 = std::make_unique<NativeBackend>();
    {
        BackendScope scope(backend2.get());
        scope.Exit();
        EXPECT_EQ(backend1.get(), GetCurrentBackend());
        SetCurrentBackend(backend2.get());
        // not recovered here because the scope has already existed
    }
    ASSERT_EQ(backend2.get(), GetCurrentBackend());
}

}  // namespace
}  // namespace xchainer
