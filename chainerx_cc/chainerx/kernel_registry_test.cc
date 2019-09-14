#include "chainerx/kernel_registry.h"

#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include "chainerx/backend.h"
#include "chainerx/context.h"
#include "chainerx/error.h"
#include "chainerx/kernel.h"
#include "chainerx/macro.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/testing/threading.h"
#include "chainerx/util.h"

namespace chainerx {
namespace {

class MyKernel : public Kernel {
public:
    virtual std::string Call(int a, const std::string& b) { return std::to_string(a) + b; }
};

class MyChildKernel : public Kernel {
public:
    virtual std::string Call(int a, const std::string& b) { return std::to_string(a) + b; }
};
class MyParentKernel : public Kernel {
public:
    virtual std::string Call(const std::string& a, float b) { return a + std::to_string(b); }
};

class MyChildKernel2 : public Kernel {
public:
    virtual std::string Call(int a, const std::string& b) { return std::to_string(a) + b; }
};

class MyParentKernel2 : public Kernel {
public:
    virtual std::string Call(const std::string& a, float b) { return a + std::to_string(b); }
};

}  // namespace

namespace internal {
CHAINERX_REGISTER_KEY_KERNEL(MyKernel, "mykernel");
CHAINERX_REGISTER_KEY_KERNEL(MyChildKernel, "mychildkernel");
CHAINERX_REGISTER_KEY_KERNEL(MyParentKernel, "myparentkernel");
CHAINERX_REGISTER_KEY_KERNEL(MyChildKernel2, "mychildkernel2");
CHAINERX_REGISTER_KEY_KERNEL(MyParentKernel2, "myparentkernel2");
}  // namespace internal

namespace {

TEST(KernelRegistryTest, KernelRegistry) {
    KernelRegistry kernel_registry{};

    kernel_registry.RegisterKernel<MyKernel, MyKernel>();

    Kernel& kernel = kernel_registry.GetKernel<MyKernel>();

    // no throw
    MyKernel& mykernel = dynamic_cast<MyKernel&>(kernel);

    EXPECT_EQ(mykernel.Call(3, " is 3"), "3 is 3");
}

TEST(KernelRegistryTest, KernelRegistryHierarchy) {
    KernelRegistry parent_kernel_registry{};
    KernelRegistry kernel_registry1{&parent_kernel_registry};
    KernelRegistry kernel_registry2{&parent_kernel_registry};

    kernel_registry1.RegisterKernel<MyChildKernel, MyChildKernel>();
    parent_kernel_registry.RegisterKernel<MyParentKernel, MyParentKernel>();

    EXPECT_THROW({ kernel_registry2.GetKernel<MyChildKernel>(); }, ChainerxError);
    EXPECT_THROW({ parent_kernel_registry.GetKernel<MyChildKernel>(); }, ChainerxError);
    // no throw
    Kernel& kernel1p = kernel_registry1.GetKernel<MyParentKernel>();
    Kernel& kernel2p = kernel_registry2.GetKernel<MyParentKernel>();
    Kernel& kernelpp = parent_kernel_registry.GetKernel<MyParentKernel>();

    Kernel& kernel = kernel_registry1.GetKernel<MyChildKernel>();
    EXPECT_EQ(&kernel1p, &kernel2p);
    EXPECT_EQ(&kernel1p, &kernelpp);
    EXPECT_NE(&kernel1p, &kernel);

    MyChildKernel& mykernel = dynamic_cast<MyChildKernel&>(kernel);
    EXPECT_EQ(mykernel.Call(3, " is 3"), "3 is 3");
}

TEST(KernelRegistryTest, KernelRegistryWithBackend) {
    // TODO(imanishi): Restore the environment variable after this test.
    SetEnv("CHAINERX_PATH", CHAINERX_TEST_DIR "/backend_testdata");
    Context ctx;
    Backend& backend0 = ctx.GetBackend("backend0");

    KernelRegistry& kernel_registry = backend0.kernel_registry();

    kernel_registry.RegisterKernel<MyKernel, MyKernel>();

    Kernel& kernel = kernel_registry.GetKernel<MyKernel>();

    // no throw
    MyKernel& mykernel = dynamic_cast<MyKernel&>(kernel);

    EXPECT_EQ(mykernel.Call(3, " is 3"), "3 is 3");

    // MyKernel should not be regsitered to the base backend: NativeBackend.
    EXPECT_THROW({ native::NativeBackend::GetGlobalKernelRegistry().GetKernel<MyKernel>(); }, ChainerxError);

    // MyKernel should not be regsitered to another Backend0 instance.
    {
        Context ctx_another;
        Backend& backend_another = ctx_another.GetBackend("backend0");
        EXPECT_THROW({ backend_another.kernel_registry().GetKernel<MyKernel>(); }, ChainerxError);
    }
}

TEST(KernelRegistryTest, KernelRegistryThreadSafe) {
    KernelRegistry parent_kernel_registry{};
    KernelRegistry kernel_registry1{&parent_kernel_registry};

    kernel_registry1.RegisterKernel<MyChildKernel, MyChildKernel>();
    parent_kernel_registry.RegisterKernel<MyParentKernel, MyParentKernel>();

    testing::RunThreads(4U, [&parent_kernel_registry, &kernel_registry1](size_t thread_index) {
        switch (thread_index) {
            case 0:
                kernel_registry1.GetKernel<MyChildKernel>();
                break;
            case 1:
                kernel_registry1.GetKernel<MyParentKernel>();
                break;
            case 2:
                kernel_registry1.RegisterKernel<MyChildKernel2, MyChildKernel2>();
                break;
            case 3:
                parent_kernel_registry.RegisterKernel<MyParentKernel2, MyParentKernel2>();
                break;
            default:
                CHAINERX_NEVER_REACH();
        }
    });
}

}  // namespace
}  // namespace chainerx
