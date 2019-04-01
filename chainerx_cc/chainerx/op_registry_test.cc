#include "chainerx/op_registry.h"

#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include "chainerx/backend.h"
#include "chainerx/context.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/op.h"
#include "chainerx/testing/threading.h"
#include "chainerx/util.h"

namespace chainerx {
namespace {

TEST(OpRegistryTest, OpRegistry) {
    OpRegistry op_registry{};

    class MyOp : public Op {
    public:
        static const char* name() { return "myop"; }
        virtual std::string Call(int a, const std::string& b) { return std::to_string(a) + b; }
    };

    op_registry.RegisterOp<MyOp, MyOp>();

    Op& op = op_registry.GetOp<MyOp>();

    // no throw
    MyOp& myop = dynamic_cast<MyOp&>(op);

    EXPECT_EQ(myop.Call(3, " is 3"), "3 is 3");
}

TEST(OpRegistryTest, OpRegistryHierarchy) {
    OpRegistry parent_op_registry{};
    OpRegistry op_registry1{&parent_op_registry};
    OpRegistry op_registry2{&parent_op_registry};

    class MyOp1 : public Op {
    public:
        static const char* name() { return "myop1"; }
        virtual std::string Call(int a, const std::string& b) { return std::to_string(a) + b; }
    };

    class MyParentOp : public Op {
    public:
        static const char* name() { return "myparentop"; }
        virtual std::string Call(const std::string& a, float b) { return a + std::to_string(b); }
    };

    op_registry1.RegisterOp<MyOp1, MyOp1>();
    parent_op_registry.RegisterOp<MyParentOp, MyParentOp>();

    EXPECT_THROW({ op_registry2.GetOp<MyOp1>(); }, ChainerxError);
    EXPECT_THROW({ parent_op_registry.GetOp<MyOp1>(); }, ChainerxError);
    // no throw
    Op& op1p = op_registry1.GetOp<MyParentOp>();
    Op& op2p = op_registry2.GetOp<MyParentOp>();
    Op& oppp = parent_op_registry.GetOp<MyParentOp>();

    Op& op = op_registry1.GetOp<MyOp1>();
    EXPECT_EQ(&op1p, &op2p);
    EXPECT_EQ(&op1p, &oppp);
    EXPECT_NE(&op1p, &op);

    MyOp1& myop = dynamic_cast<MyOp1&>(op);
    EXPECT_EQ(myop.Call(3, " is 3"), "3 is 3");
}

TEST(OpRegistryTest, OpRegistryWithBackend) {
    // TODO(imanishi): Restore the environment variable after this test.
    SetEnv("CHAINERX_PATH", CHAINERX_TEST_DIR "/backend_testdata");
    Context ctx;
    Backend& backend0 = ctx.GetBackend("backend0");

    OpRegistry& op_registry = backend0.op_registry();

    class MyOp : public Op {
    public:
        static const char* name() { return "myop"; }
        virtual std::string Call(int a, const std::string& b) { return std::to_string(a) + b; }
    };

    op_registry.RegisterOp<MyOp, MyOp>();

    Op& op = op_registry.GetOp<MyOp>();

    // no throw
    MyOp& myop = dynamic_cast<MyOp&>(op);

    EXPECT_EQ(myop.Call(3, " is 3"), "3 is 3");

    // MyOp should not be regsitered to the base backend: NativeBackend.
    EXPECT_THROW({ native::NativeBackend::GetGlobalOpRegistry().GetOp<MyOp>(); }, ChainerxError);

    // MyOp should not be regsitered to another Backend0 instance.
    {
        Context ctx_another;
        Backend& backend_another = ctx_another.GetBackend("backend0");
        EXPECT_THROW({ backend_another.op_registry().GetOp<MyOp>(); }, ChainerxError);
    }
}

TEST(OpRegistryTest, OpRegistryThreadSafe) {
    OpRegistry parent_op_registry{};
    OpRegistry op_registry1{&parent_op_registry};

    class MyOp1 : public Op {
    public:
        static const char* name() { return "myop1"; }
        virtual std::string Call(int a, const std::string& b) { return std::to_string(a) + b; }
    };

    class MyParentOp : public Op {
    public:
        static const char* name() { return "myparentop"; }
        virtual std::string Call(const std::string& a, float b) { return a + std::to_string(b); }
    };

    class MyOp2 : public Op {
    public:
        static const char* name() { return "myop2"; }
        virtual std::string Call(int a, const std::string& b) { return std::to_string(a) + b; }
    };

    class MyParentOp2 : public Op {
    public:
        static const char* name() { return "myparentop2"; }
        virtual std::string Call(const std::string& a, float b) { return a + std::to_string(b); }
    };

    op_registry1.RegisterOp<MyOp1, MyOp1>();
    parent_op_registry.RegisterOp<MyParentOp, MyParentOp>();

    testing::RunThreads(4U, [&parent_op_registry, &op_registry1](size_t thread_index) {
        switch (thread_index) {
            case 0:
                op_registry1.GetOp<MyOp1>();
                break;
            case 1:
                op_registry1.GetOp<MyParentOp>();
                break;
            case 2:
                op_registry1.RegisterOp<MyOp2, MyOp2>();
                break;
            case 3:
                parent_op_registry.RegisterOp<MyParentOp2, MyParentOp2>();
                break;
            default:
                CHAINERX_NEVER_REACH();
        }
    });
}

}  // namespace
}  // namespace chainerx
